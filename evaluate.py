#!/usr/bin/python
# -*- encoding: utf-8 -*-


from email.policy import strict
import sys

sys.path.insert(0, '.')
import os
import os.path as osp

sys.path.append('/root/autodl-tmp/project/BiSeNet')  # 指定绝对路径防止出现无法import文件

import logging
import argparse
import math
from tabulate import tabulate

from tqdm import tqdm
import numpy as np
import cv2
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist

from lib.models import model_factory
from configs import set_cfg_from_file
from lib.logger import setup_logger
from lib.get_dataloader import get_data_loader, get_city_loader
from lib.city_to_cam import Cityid_to_Camid
from lib.a2d2_to_cam import a2d2_to_Camid
from lib.class_remap import ClassRemap
from tools.configer import Configer

CITY_ID = 0
CAM_ID = 1

def get_round_size(size, divisor=32):
    return [math.ceil(el / divisor) * divisor for el in size]

class MscEvalV0(object):

    def __init__(self, scales=(0.5, ), flip=False, ignore_label=255):
        self.scales = scales
        self.flip = flip
        self.ignore_label = ignore_label

    def __call__(self, net, dl, n_classes, dataset_id):
        ## evaluate
        hist = torch.zeros(n_classes, n_classes).cuda().detach()
        if dist.is_initialized() and dist.get_rank() != 0:
            diter = enumerate(dl)
        else:
            diter = enumerate(tqdm(dl))
        for i, (imgs, label) in diter:
            N, _, H, W = label.shape
            label = label.squeeze(1).cuda()
            size = label.size()[-2:]
            probs = torch.zeros(
                    (N, n_classes, H, W), dtype=torch.float32).cuda().detach()

            for scale in self.scales:
                sH, sW = int(scale * H), int(scale * W)
                sH, sW = get_round_size((sH, sW))
                im_sc = F.interpolate(imgs, size=(sH, sW),
                        mode='bilinear', align_corners=True)

                im_sc = im_sc.cuda()
                
                logits = net(im_sc, dataset=dataset_id)[0]
                logits = F.interpolate(logits, size=size,
                        mode='bilinear', align_corners=True)
                probs += torch.softmax(logits, dim=1)
                if self.flip:
                    im_sc = torch.flip(im_sc, dims=(3, ))
                    logits = net(im_sc, dataset=dataset_id)[0]
                    logits = torch.flip(logits, dims=(3, ))
                    logits = F.interpolate(logits, size=size,
                            mode='bilinear', align_corners=True)
                    probs += torch.softmax(logits, dim=1)
            preds = torch.argmax(probs, dim=1)
            

            keep = label != self.ignore_label
            hist += torch.tensor(np.bincount(
                label.cpu().numpy()[keep.cpu().numpy()] * n_classes + preds.cpu().numpy()[keep.cpu().numpy()],
                minlength=n_classes ** 2
            )).cuda().view(n_classes, n_classes)
        if dist.is_initialized():
            dist.all_reduce(hist, dist.ReduceOp.SUM)
        ious = hist.diag() / (hist.sum(dim=0) + hist.sum(dim=1) - hist.diag())
        print(ious)
        miou = np.nanmean(ious.detach().cpu().numpy())
        return miou.item()

class MscEvalV0_Contrast(object):

    def __init__(self, configer, scales=(0.5, ), flip=False, ignore_label=255):
        self.configer = configer
        self.num_unify_classes = self.configer.get('num_unify_classes')
        self.class_Remaper = ClassRemap(configer=self.configer)
        self.scales = scales
        self.flip = flip
        self.ignore_label = ignore_label

    def __call__(self, net, dl, n_classes, dataset_id):
        ## evaluate
        hist = torch.zeros(n_classes, n_classes).cuda().detach()
        if dist.is_initialized() and dist.get_rank() != 0:
            diter = enumerate(dl)
        else:
            diter = enumerate(tqdm(dl))
        for i, (imgs, label) in diter:
            N, _, H, W = label.shape

            label = label.squeeze(1).cuda()
            size = label.size()[-2:]
            probs = torch.zeros(
                    (N, self.num_unify_classes, H, W), dtype=torch.float32).cuda().detach()

            for scale in self.scales:
                sH, sW = int(scale * H), int(scale * W)
                sH, sW = get_round_size((sH, sW))
                im_sc = F.interpolate(imgs, size=(sH, sW),
                        mode='bilinear', align_corners=True)

                im_sc = im_sc.cuda()
                
                logits = net(im_sc)
                logits = F.interpolate(logits, size=size,
                        mode='bilinear', align_corners=True)
                probs += torch.softmax(logits, dim=1)
                if self.flip:
                    im_sc = torch.flip(im_sc, dims=(3, ))
                    logits = net(im_sc)
                    logits = torch.flip(logits, dims=(3, ))
                    logits = F.interpolate(logits, size=size,
                            mode='bilinear', align_corners=True)
                    probs += torch.softmax(logits, dim=1)
            preds = torch.argmax(probs, dim=1)

            # if dataset_id == CITY_ID:
            #     # CityScapes数据集一一对应不需要逆映射
            #     preds = self.class_Remaper.ReverseSegRemap(preds, dataset_id)

            if dataset_id == CAM_ID:
                # CityScapes数据集一一对应不需要逆映射
                preds = self.class_Remaper.ReverseSegRemap(preds, dataset_id)

            keep = label != self.ignore_label

            hist += torch.tensor(np.bincount(
                label.cpu().numpy()[keep.cpu().numpy()] * n_classes + preds.cpu().numpy()[keep.cpu().numpy()],
                minlength=n_classes ** 2
            )).cuda().view(n_classes, n_classes)
                
        if dist.is_initialized():
            dist.all_reduce(hist, dist.ReduceOp.SUM)
        ious = hist.diag() / (hist.sum(dim=0) + hist.sum(dim=1) - hist.diag())
        print(ious)
        miou = np.nanmean(ious.detach().cpu().numpy())
        return miou.item()


class MscEvalCrop(object):

    def __init__(
        self,
        cropsize=1024,
        cropstride=2./3,
        flip=True,
        scales=[0.5, 0.75, 1, 1.25, 1.5, 1.75],
        lb_ignore=255,
    ):
        self.scales = scales
        self.ignore_label = lb_ignore
        self.flip = flip
        self.distributed = dist.is_initialized()

        self.cropsize = cropsize if isinstance(cropsize, (list, tuple)) else (cropsize, cropsize)
        self.cropstride = cropstride


    def pad_tensor(self, inten):
        N, C, H, W = inten.size()
        cropH, cropW = self.cropsize
        if cropH < H and cropW < W: return inten, [0, H, 0, W]
        padH, padW = max(cropH, H), max(cropW, W)
        outten = torch.zeros(N, C, padH, padW).cuda()
        outten.requires_grad_(False)
        marginH, marginW = padH - H, padW - W
        hst, hed = marginH // 2, marginH // 2 + H
        wst, wed = marginW // 2, marginW // 2 + W
        outten[:, :, hst:hed, wst:wed] = inten
        return outten, [hst, hed, wst, wed]


    def eval_chip(self, net, crop):
        prob = net(crop)[0].softmax(dim=1)
        if self.flip:
            crop = torch.flip(crop, dims=(3,))
            prob += net(crop)[0].flip(dims=(3,)).softmax(dim=1)
            prob = torch.exp(prob)
        return prob


    def crop_eval(self, net, im, n_classes):
        cropH, cropW = self.cropsize
        stride_rate = self.cropstride
        im, indices = self.pad_tensor(im)
        N, C, H, W = im.size()

        strdH = math.ceil(cropH * stride_rate)
        strdW = math.ceil(cropW * stride_rate)
        n_h = math.ceil((H - cropH) / strdH) + 1
        n_w = math.ceil((W - cropW) / strdW) + 1
        prob = torch.zeros(N, n_classes, H, W).cuda()
        prob.requires_grad_(False)
        for i in range(n_h):
            for j in range(n_w):
                stH, stW = strdH * i, strdW * j
                endH, endW = min(H, stH + cropH), min(W, stW + cropW)
                stH, stW = endH - cropH, endW - cropW
                chip = im[:, :, stH:endH, stW:endW]
                prob[:, :, stH:endH, stW:endW] += self.eval_chip(net, chip)
        hst, hed, wst, wed = indices
        prob = prob[:, :, hst:hed, wst:wed]
        return prob


    def scale_crop_eval(self, net, im, scale, n_classes):
        N, C, H, W = im.size()
        new_hw = [int(H * scale), int(W * scale)]
        im = F.interpolate(im, new_hw, mode='bilinear', align_corners=True)
        prob = self.crop_eval(net, im, n_classes)
        prob = F.interpolate(prob, (H, W), mode='bilinear', align_corners=True)
        return prob


    @torch.no_grad()
    def __call__(self, net, dl, n_classes):
        dloader = dl if self.distributed and not dist.get_rank() == 0 else tqdm(dl)

        hist = torch.zeros(n_classes, n_classes).cuda().detach()
        hist.requires_grad_(False)
        for i, (imgs, label) in enumerate(dloader):
            imgs = imgs.cuda()
            label = label.squeeze(1).cuda()
            N, H, W = label.shape

            probs = torch.zeros((N, n_classes, H, W)).cuda()
            probs.requires_grad_(False)
            for sc in self.scales:
                probs += self.scale_crop_eval(net, imgs, sc, n_classes)
            torch.cuda.empty_cache()
            preds = torch.argmax(probs, dim=1)

            keep = label != self.ignore_label
            hist += torch.bincount(
                label[keep] * n_classes + preds[keep],
                minlength=n_classes ** 2
                ).view(n_classes, n_classes)

        if self.distributed:
            dist.all_reduce(hist, dist.ReduceOp.SUM)
        ious = hist.diag() / (hist.sum(dim=0) + hist.sum(dim=1) - hist.diag())
        miou = np.nanmean(ious.detach().cpu().numpy())
        return miou.item()

# 修改后用于多数据集
class MscEvalCrop_cdcl(object):

    def __init__(
        self,
        cropsize=[1024, 1024], # 两个数据集的cropsize
        cropstride=2./3,
        flip=True,
        scales=[0.5, 0.75, 1, 1.25, 1.5, 1.75],
        lb_ignore=255,
    ):
        self.scales = scales
        self.ignore_label = lb_ignore
        self.flip = flip
        self.distributed = dist.is_initialized()

        # 修改cropsize为多数据集
        self.cropsize = [c_size if isinstance(c_size, (list, tuple)) else (c_size, c_size) for c_size in cropsize]
        self.cropstride = cropstride


    def pad_tensor(self, inten):
        N, C, H, W = inten.size()
        # 暂用，待完善
        cropH, cropW = self.cropsize[0]
        if cropH < H and cropW < W: return inten, [0, H, 0, W]
        padH, padW = max(cropH, H), max(cropW, W)
        outten = torch.zeros(N, C, padH, padW).cuda()
        outten.requires_grad_(False)
        marginH, marginW = padH - H, padW - W
        hst, hed = marginH // 2, marginH // 2 + H
        wst, wed = marginW // 2, marginW // 2 + W
        outten[:, :, hst:hed, wst:wed] = inten
        return outten, [hst, hed, wst, wed]


    def eval_chip(self, net, crop):
        # 修改后用于多数据集
        prob = net(*crop)[0]
        prob = [pb.softmax(dim=1) for pb in prob]
        if self.flip:
            crop = [torch.flip(cp, dims=(3,)) for cp in crop]
            out_tmp = net(*crop)[0]
            out_tmp = [ot.flip(dims=(3,)).softmax(dim=1) for ot in out_tmp]
            prob = [pb + ot for pb, ot in zip(prob, out_tmp)]
            prob = [torch.exp(pb) for pb in prob]
        return prob


    def crop_eval(self, net, im, n_classes):
        # 修改后用于多数据集
        cropH, cropW = [[c_size[j] for c_size in self.cropsize] for j in range(0, 2)]
        stride_rate = self.cropstride
        indices = []
        im_tmp = []
        for j in range(0, len(im)):
            im_t, in_t = self.pad_tensor(im[j])
            im_tmp.append(im_t)
            indices.append(in_t)
        im = im_tmp
        N, C, H, W = [[img.size()[j] for img in im] for j in range(0, 4)]

        strdH = [math.ceil(ch * stride_rate) for ch in cropH]
        strdW = [math.ceil(cw * stride_rate) for cw in cropW]
        n_h = [math.ceil((h - ch) / sh) + 1 for h, ch, sh in zip(H, cropH, strdH)]
        n_w = [math.ceil((w - cw) / sw) + 1 for w, cw, sw in zip(W, cropW, strdW)]
        prob = [(torch.zeros((N[j], n_classes[j], H[j], W[j])).cuda()) for j in range(0, len(N))]
        for j in range(0, len(prob)):
            prob[j].requires_grad_(False)
        # 数据集的cropsize不同时可能有问题，暂用，待完善
        for i in range(n_h[0]):
            for j in range(n_w[0]):
                stH, stW = [sh * i for sh in strdH], [sw * j for sw in strdW]
                endH = [min(h, sth + ch) for h, sth, ch in zip(H, stH, cropH)]
                endW = [min(w, stw + cw) for w, stw, cw in zip(W, stW, cropW)]
                stH = [eh - ch for eh, ch in zip(endH, cropH)]
                stW = [ew - cw for ew, cw in zip(endW, cropW)]
                chip = [im[k][:, :, stH[k]:endH[k], stW[k]:endW[k]] for k in range(0, len(im))]
                e_chip = self.eval_chip(net, chip)
                for k in range(0, len(prob)):
                    prob[k][:, :, stH[k]:endH[k], stW[k]:endW[k]] += e_chip[k]
        hst, hed, wst, wed = [[ind[j] for ind in indices] for j in range(0, 4)]
        prob = [prob[j][:, :, hst[j]:hed[j], wst[j]:wed[j]] for j in range(0, len(prob))]
        return prob


    def scale_crop_eval(self, net, im, scale, n_classes):
        # 修改后用于多数据集
        N, C, H, W = [[img.size()[j] for img in im] for j in range(0, 4)]
        new_hw = [[int(h * scale) for h in H], [int(w * scale) for w in W]]
        im = [F.interpolate(im[j], (new_hw[0][j], new_hw[1][j]), mode='bilinear', align_corners=True) for j in range(0, len(im))]
        prob = self.crop_eval(net, im, n_classes)
        prob = [F.interpolate(prob[j], (H[j], W[j]), mode='bilinear', align_corners=True) for j in range(0, len(prob))]
        return prob


    @torch.no_grad()
    def __call__(self, net, dl, n_classes):
        # 修改后用于多数据集
        diter = [iter(d) for d in dl]
        hist = [torch.zeros(n_c, n_c).cuda().detach() for n_c in n_classes]
        # a2d2的验证集大约为4000，cityscapes的为500，cityscapes的多次计算
        num_imgs = 100
        batch_size = 2
        for i in range(0, num_imgs):
            if (i + 1) % 50 == 0:
                print('iter:', i + 1)
            imgs = []
            label = []
            for j in range(0, len(diter)):
                try:
                    im_ds, lb_ds = next(diter[j])
                    if not im_ds.size()[0] == batch_size:
                        raise StopIteration
                except StopIteration:
                    diter[j] = iter(dl[j])
                    im_ds, lb_ds = next(diter[j])
                imgs.append(im_ds.cuda())
                label.append(lb_ds)
            N, _, H, W = [[lb.size()[j] for lb in label] for j in range(0, 4)]
            label = [lb.squeeze(1).cuda() for lb in label]
            probs = [(torch.zeros((N[j], n_classes[j], H[j], W[j]),
                                  dtype=torch.float32).cuda()) for j in range(0, len(N))]
            for j in range(len(probs)):
                probs[j].requires_grad_(False)
            for sc in self.scales:
                s_c_e = self.scale_crop_eval(net, imgs, sc, n_classes)
                probs = [pb + sce for pb, sce in zip(probs, s_c_e)]
            torch.cuda.empty_cache()
            preds = [torch.argmax(pb, dim=1) for pb in probs]

            keep = [lb != self.ignore_label for lb in label]
            # bincount = [torch.bincount(
            #     label[j][keep[j]] * n_classes[j] + preds[j][keep[j]],
            #     minlength=n_classes[j] ** 2
            # ).view(n_classes[j], n_classes[j]) for j in range(0, len(label))]
            # 可能由于某些tensor在显存不连续导致这一步计算速度很慢，放到numpy上计算
            bincount = [torch.tensor(np.bincount(
                label[j].cpu().numpy()[keep[j].cpu().numpy()] * n_classes[j] + preds[j].cpu().numpy()[
                    keep[j].cpu().numpy()],
                minlength=n_classes[j] ** 2
            )).cuda().view(n_classes[j], n_classes[j]) for j in range(0, len(label))]
            hist = [h + b for h, b in zip(hist, bincount)]

        if self.distributed:
            for j in range(0, len(hist)):
                dist.all_reduce(hist[j], dist.ReduceOp.SUM)
        ious = [h.diag() / (h.sum(dim=0) + h.sum(dim=1) - h.diag()) for h in hist]
        miou = [np.nanmean(i.detach().cpu().numpy()) for i in ious]
        return [mi.item() for mi in miou]

@torch.no_grad()
def eval_model(cfg, net):
    org_aux = net.aux_mode
    net.aux_mode = 'eval'

    is_dist = dist.is_initialized()
    dl = get_data_loader(cfg, mode='val', distributed=is_dist)
    net.eval()

    heads, mious = [], []
    logger = logging.getLogger()

    single_scale = MscEvalV0((1., ), False)
    mIOU = single_scale(net, dl, cfg.n_cats)
    heads.append('single_scale')
    mious.append(mIOU)
    logger.info('single mIOU is: %s\n', mIOU)

    # single_crop = MscEvalCrop(
    #     cropsize=cfg.eval_crop,
    #     cropstride=2. / 3,
    #     flip=False,
    #     scales=(1., ),
    #     lb_ignore=255,
    # )
    # mIOU = single_crop(net, dl, cfg.n_cats)
    # heads.append('single_scale_crop')
    # mious.append(mIOU)
    # logger.info('single scale crop mIOU is: %s\n', mIOU)

    # ms_flip = MscEvalV0(cfg.eval_scales, True)
    # mIOU = ms_flip(net, dl, cfg.n_cats)
    # heads.append('ms_flip')
    # mious.append(mIOU)
    # logger.info('ms flip mIOU is: %s\n', mIOU)

    # ms_flip_crop = MscEvalCrop(
    #     cropsize=cfg.eval_crop,
    #     cropstride=2. / 3,
    #     flip=True,
    #     scales=cfg.eval_scales,
    #     lb_ignore=255,
    # )
    # mIOU = ms_flip_crop(net, dl, cfg.n_cats)
    # heads.append('ms_flip_crop')
    # mious.append(mIOU)
    # logger.info('ms crop mIOU is: %s\n', mIOU)

    net.aux_mode = org_aux
    return heads, mious

def eval_model_cdcl(cfg_a2d2, cfg_city, net):
    # 修改后用于多数据集
    org_aux = net.aux_mode
    net.aux_mode = 'eval'

    is_dist = dist.is_initialized()
    dl_a2d2 = get_data_loader(cfg_a2d2, mode='val', distributed=is_dist)
    dl_city = get_data_loader(cfg_city, mode='val', distributed=is_dist)
    net.eval()

    heads, mious = [], []
    logger = logging.getLogger()

    single_scale = MscEvalV0_cdcl((0.5, ), False)
    mIOU = single_scale(net, [dl_a2d2, dl_city], [cfg_a2d2.n_cats, cfg_city.n_cats])
    heads.append('single_scale')
    mious.append(mIOU)
    logger.info('A2D2 single mIOU is: %s\nCityScapes single mIOU is: %s\n', mIOU[0], mIOU[1])

    # single_crop = MscEvalCrop_cdcl(
    #     cropsize=[cfg_a2d2.eval_crop, cfg_city.eval_crop],
    #     cropstride=2. / 3,
    #     flip=False,
    #     scales=(1., ),
    #     lb_ignore=255,
    # )
    # mIOU = single_crop(net, [dl_a2d2, dl_city], [cfg_a2d2.n_cats, cfg_city.n_cats])
    # heads.append('single_scale_crop')
    # mious.append(mIOU)
    # logger.info('A2D2 single scale crop mIOU is: %s\nCityScapes single scale crop mIOU is: %s\n', mIOU[0], mIOU[1])

    # ms_flip = MscEvalV0_cdcl(cfg_a2d2.eval_scales, True)
    # mIOU = ms_flip(net, [dl_a2d2, dl_city], [cfg_a2d2.n_cats, cfg_city.n_cats])
    # heads.append('ms_flip')
    # mious.append(mIOU)
    # logger.info('A2D2 ms flip mIOU is: %s\nCityScapes ms flip mIOU is: %s\n', mIOU[0], mIOU[1])

    # ms_flip_crop = MscEvalCrop_cdcl(
    #     cropsize=[cfg_a2d2.eval_crop, cfg_city.eval_crop],
    #     cropstride=2. / 3,
    #     flip=True,
    #     scales=cfg_a2d2.eval_scales,
    #     lb_ignore=255,
    # )
    # mIOU = ms_flip_crop(net, [dl_a2d2, dl_city], [cfg_a2d2.n_cats, cfg_city.n_cats])
    # heads.append('ms_flip_crop')
    # mious.append(mIOU)
    # logger.info('A2D2 ms crop mIOU is: %s\nCityScapes ms crop mIOU is: %s\n', mIOU)

    net.aux_mode = org_aux
    return heads, mious

def evaluate(cfg, weight_pth):
    logger = logging.getLogger()

    ## model
    logger.info('setup and restore model\nmodel: %s', weight_pth)
    net = model_factory[cfg.model_type](cfg.n_cats)
    net.load_state_dict(torch.load(weight_pth, map_location='cpu'))
    net.cuda()

    is_dist = dist.is_initialized()
    if is_dist:
        local_rank = dist.get_rank()
        net = nn.parallel.DistributedDataParallel(
            net,
            device_ids=[local_rank, ],
            output_device=local_rank
        )

    ## evaluator
    heads, mious = eval_model(cfg, net.module)
    logger.info(tabulate([mious, ], headers=heads, tablefmt='orgtbl'))

def evaluate_cdcl(cfg_a2d2, cfg_city, cfg_cam, weight_pth):
    # 修改后用于多数据集
    logger = logging.getLogger()

    ## model
    logger.info('setup and restore model\nmodel: %s', weight_pth)
    net = model_factory[cfg_a2d2.model_type](cfg_a2d2.n_cats, 'eval', 2, cfg_city.n_cats)
    net.load_state_dict(torch.load(weight_pth, map_location='cpu'), strict=True)
    # net.load_state_dict(torch.load('res/bga_precise_bn.pth', map_location='cpu'), strict=False)
    # # for precise bn
    # net.segment.load_state_dict(torch.load('res/segment_precise_bn_40.pth', map_location='cpu'), strict=False)
    # net.detail.load_state_dict(torch.load('res/detail_precise_bn.pth', map_location='cpu'), strict=False)
    net.eval()
    net.cuda()

    is_dist = dist.is_initialized()
    if is_dist:
        local_rank = dist.get_rank()
        net = nn.parallel.DistributedDataParallel(
            net,
            device_ids=[local_rank, ],
            output_device=local_rank
        )

    ## evaluator
    # heads, mious = eval_model_cdcl(cfg_a2d2, cfg_city, net.module)
    # heads, mious = eval_model(cfg_city, net)
    heads, mious = eval_model(cfg_cam, net)
    # # 输出miou表格
    # miou_tab = []
    # for i in range(len(mious[0])):
    #     tab = []
    #     for j in range(len(mious)):
    #         tab.append(mious[j][i])
    #     miou_tab.append(tab)
    # miou_tab[0] = ['A2D2'] + miou_tab[0]
    # miou_tab[1] = ['CityScapes'] + miou_tab[1]
    # heads = ['dataset'] + heads
    # logger.info(tabulate(miou_tab, headers=heads, tablefmt='orgtbl'))

# def evaluate_cdcl(cfg_a2d2, cfg_city, weight_pth):
#     # 修改后用于多数据集
#     logger = logging.getLogger()

#     ## model
#     logger.info('setup and restore model\nmodel: %s', weight_pth)
#     net = model_factory[cfg_a2d2.model_type](cfg_a2d2.n_cats, 'val', 2, cfg_city.n_cats)
#     net.load_state_dict(torch.load(weight_pth, map_location='cpu'), strict=False)
#     net.cuda()

#     is_dist = dist.is_initialized()
#     if is_dist:
#         local_rank = dist.get_rank()
#         net = nn.parallel.DistributedDataParallel(
#             net,
#             device_ids=[local_rank, ],
#             output_device=local_rank
#         )

#     ## evaluator
#     # heads, mious = eval_model_cdcl(cfg_a2d2, cfg_city, net.module)
#     heads, mious = eval_model(cfg_city, net)
#     logger.info(tabulate([mious, ], headers=heads, tablefmt='orgtbl'))
#     # # 输出miou表格
#     # miou_tab = []
#     # for i in range(len(mious[0])):
#     #     tab = []
#     #     for j in range(len(mious)):
#     #         tab.append(mious[j][i])
#     #     miou_tab.append(tab)
#     # miou_tab[0] = ['A2D2'] + miou_tab[0]
#     # miou_tab[1] = ['CityScapes'] + miou_tab[1]
#     # heads = ['dataset'] + heads
#     # logger.info(tabulate(miou_tab, headers=heads, tablefmt='orgtbl'))

@torch.no_grad()
def eval_model_contrast(configer, net):
    org_aux = net.aux_mode
    net.aux_mode = 'eval'

    is_dist = dist.is_initialized()
    
    # cfg_city = set_cfg_from_file(configer.get('dataset1'))
    # cfg_cam  = set_cfg_from_file(configer.get('dataset2'))

    n_datasets = configer.get("n_datasets")

    # dl_cam = get_data_loader(cfg_cam, mode='val', distributed=is_dist)
    dl_city, dl_cam = get_data_loader(configer, aux_mode='eval', distributed=is_dist)
    net.eval()
    # net.train()

    heads, mious = [], []
    logger = logging.getLogger()

    single_scale = MscEvalV0_Contrast(configer, (1., ), False)
    
    mIOU_city = single_scale(net, dl_city, 19, CITY_ID)
    mIOU_cam = single_scale(net, dl_cam, configer.get('dataset2', 'n_cats'), CAM_ID)

    heads.append('single_scale')
    mious.append(mIOU_cam)
    mious.append(mIOU_city)
    logger.info('Cam single mIOU is: %s\nCityScapes single mIOU is: %s\n', mIOU_cam, mIOU_city)

    net.aux_mode = org_aux
    return heads, mious

@torch.no_grad()
def eval_model_contrast_single(configer, net):
    org_aux = net.aux_mode
    net.aux_mode = 'eval'

    is_dist = dist.is_initialized()
    
    # cfg_city = set_cfg_from_file(configer.get('dataset1'))
    # cfg_cam  = set_cfg_from_file(configer.get('dataset2'))

    # n_datasets = configer.get("n_datasets")

    # dl_cam = get_data_loader(cfg_cam, mode='val', distributed=is_dist)
    dl_city = get_data_loader(configer, aux_mode='eval', distributed=is_dist)[0]
    net.eval()

    heads, mious = [], []
    logger = logging.getLogger()

    single_scale = MscEvalV0_Contrast(configer, (1., ), False)
    
    mIOU_city = single_scale(net, dl_city, configer.get('dataset1', 'n_cats'), CITY_ID)
    # mIOU_cam = single_scale(net, dl_cam, configer.get('dataset2', 'n_cats'), CAM_ID)

    heads.append('single_scale')
    # mious.append(mIOU_cam)
    mious.append(mIOU_city)
    logger.info('CityScapes single mIOU is: %s\n', mIOU_city)

    net.aux_mode = org_aux
    return heads, mious

@torch.no_grad()
def eval_model_aux(configer, net):
    org_aux = net.aux_mode
    net.aux_mode = 'eval'

    is_dist = dist.is_initialized()
    
    # cfg_city = set_cfg_from_file(configer.get('dataset1'))
    # cfg_cam  = set_cfg_from_file(configer.get('dataset2'))

    # dl_cam = get_data_loader(cfg_cam, mode='val', distributed=is_dist)
    _, dl_cam = get_data_loader(configer, aux_mode='eval', distributed=is_dist)
    dl_city = get_city_loader(configer, aux_mode='eval', distributed=is_dist)
    
    net.eval()

    heads, mious = [], []
    logger = logging.getLogger()

    single_scale = MscEvalV0((1., ), False)
    
    mIOU_city = single_scale(net, dl_city, configer.get('dataset1', 'n_cats'), CITY_ID)
    mIOU_cam = single_scale(net, dl_cam, configer.get('dataset2', 'n_cats'), CAM_ID)

    heads.append('single_scale')
    mious.append(mIOU_cam)
    mious.append(mIOU_city)
    logger.info('Cam single mIOU is: %s\nCityScapes single mIOU is: %s\n', mIOU_cam, mIOU_city)

    net.aux_mode = org_aux
    return heads, mious

@torch.no_grad()
def eval_model_emb(configer, net):
    org_aux = net.aux_mode
    net.aux_mode = 'pred_by_emb'

    is_dist = dist.is_initialized()
    
    # cfg_city = set_cfg_from_file(configer.get('dataset1'))
    # cfg_cam  = set_cfg_from_file(configer.get('dataset2'))

    # dl_cam = get_data_loader(cfg_cam, mode='val', distributed=is_dist)
    _, dl_cam = get_data_loader(configer, aux_mode='eval', distributed=is_dist)
    dl_city = get_city_loader(configer, aux_mode='eval', distributed=is_dist)
    
    net.eval()

    heads, mious = [], []
    logger = logging.getLogger()

    single_scale = MscEvalV0_Contrast(configer, (1., ), False)
    
    mIOU_city = single_scale(net, dl_city, 19, CITY_ID)
    mIOU_cam = single_scale(net, dl_cam, configer.get('dataset2', 'n_cats'), CAM_ID)

    heads.append('single_scale')
    mious.append(mIOU_city)
    mious.append(mIOU_cam)

    logger.info('Cam single mIOU is: %s\nCityScapes single mIOU is: %s\n', mIOU_cam, mIOU_city)

    net.aux_mode = org_aux
    return heads, mious


def parse_args():
    parse = argparse.ArgumentParser()
    parse.add_argument('--local_rank', dest='local_rank', type=int, default=-1,)
    parse.add_argument('--port', dest='port', type=int, default=16745,)
    parse.add_argument('--finetune_from', type=str, default=None,)
    parse.add_argument('--config', dest='config', type=str, default='configs/bisenetv2_eval.json',)
    return parse.parse_args()


def main():
    # 修改后用于多数据集
    args = parse_args()
    configer = Configer(configs=args.config)


    # if not args.local_rank == -1:
    #     torch.cuda.set_device(args.local_rank)
    #     dist.init_process_group(backend='nccl',
    #     init_method='tcp://127.0.0.1:{}'.format(args.port),
    #     world_size=torch.cuda.device_count(),
    #     rank=args.local_rank
    # )
    if not osp.exists(configer.get('res_save_pth')): os.makedirs(configer.get('res_save_pth'))
    setup_logger('{}-eval'.format(configer.get('model_name')), configer.get('res_save_pth'))
    
    logger = logging.getLogger()
    net = model_factory[configer.get('model_name')](configer)
    state = torch.load(configer.get("train", "finetune_from"), map_location='cpu')
    net.load_state_dict(state, strict=False)
    
    net.cuda()
    net.aux_mode = 'eval'
    # net.eval()
    
    heads, mious = eval_model_contrast(configer, net)
    
    # heads, mious = eval_model_emb(configer, net)
    logger.info(tabulate([mious, ], headers=heads, tablefmt='orgtbl'))
    
    

if __name__ == "__main__":
    main()
