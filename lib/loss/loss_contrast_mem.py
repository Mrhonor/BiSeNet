from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from abc import ABC
import this

import torch
import torch.nn as nn
import torch.nn.functional as F

from lib.loss.loss_helper import FSAuxCELoss, FSRMILoss, FSCELoss, FSCELOVASZLoss
from tools.logger import Logger as Log


class PixelContrastLoss(nn.Module, ABC):
    def __init__(self, configer):
        super(PixelContrastLoss, self).__init__()

        self.configer = configer
        self.temperature = self.configer.get('contrast', 'temperature')
        self.base_temperature = self.configer.get('contrast', 'base_temperature')

        self.ignore_label = self.configer.get('loss', 'ignore_index')

        self.max_samples = self.configer.get('contrast', 'max_samples')
        self.max_views = self.configer.get('contrast', 'max_views')

    def _hard_anchor_sampling(self, X, y_hat, y):
        batch_size, feat_dim = X.shape[0], X.shape[-1]

        classes = []
        total_classes = 0
        for ii in range(batch_size):
            this_y = y_hat[ii]
            this_classes = torch.unique(this_y)
            
            this_classes_ignore = [x for x in this_classes if x != self.ignore_label]
            this_classes = [x for x in this_classes_ignore if (this_y == x).nonzero().shape[0] > self.max_views]
            
            classes.append(this_classes)
            total_classes += len(this_classes)

        if total_classes == 0:
            print("this_y: ", this_y)
            print("this_classes: ", this_classes)
            print("this_classes_ignore: ", this_classes_ignore)
            print("batch_size: ", batch_size)
            return None, None

        ## 每个锚点保留个数
        n_view = self.max_samples // total_classes
        n_view = min(n_view, self.max_views)
        
        X_ = torch.zeros((total_classes, n_view, feat_dim), dtype=torch.float).cuda()
        y_ = torch.zeros(total_classes, dtype=torch.float).cuda()

        X_ptr = 0
        for ii in range(batch_size):
            this_y_hat = y_hat[ii]
            this_y = y[ii]
            this_classes = classes[ii]

            for cls_id in this_classes:
                hard_indices = ((this_y_hat == cls_id) & (this_y != cls_id)).nonzero()
                easy_indices = ((this_y_hat == cls_id) & (this_y == cls_id)).nonzero()

                num_hard = hard_indices.shape[0]
                num_easy = easy_indices.shape[0]

                if num_hard >= n_view / 2 and num_easy >= n_view / 2:
                    num_hard_keep = n_view // 2
                    num_easy_keep = n_view - num_hard_keep
                elif num_hard >= n_view / 2:
                    num_easy_keep = num_easy
                    num_hard_keep = n_view - num_easy_keep
                elif num_easy >= n_view / 2:
                    num_hard_keep = num_hard
                    num_easy_keep = n_view - num_hard_keep
                else:
                    Log.info('this shoud be never touched! {} {} {}'.format(num_hard, num_easy, n_view))
                    raise Exception

                perm = torch.randperm(num_hard)
                hard_indices = hard_indices[perm[:num_hard_keep]]
                perm = torch.randperm(num_easy)
                easy_indices = easy_indices[perm[:num_easy_keep]]
                indices = torch.cat((hard_indices, easy_indices), dim=0)

                X_[X_ptr, :, :] = X[ii, indices, :].squeeze(1)
                y_[X_ptr] = cls_id
                X_ptr += 1

        return X_, y_

    def _sample_negative(self, Q):
        class_num, feat_size = Q.shape

        X_ = torch.zeros((class_num, feat_size)).float().cuda()
        y_ = torch.zeros((class_num, 1)).float().cuda()
        sample_ptr = 0
        for ii in range(class_num):
            # if ii == 0: continue
            this_q = Q[ii, :]

            X_[sample_ptr, ...] = this_q
            y_[sample_ptr, ...] = ii
            sample_ptr += 1

        return X_, y_

    def _contrastive(self, X_anchor, y_anchor, queue=None):
        if X_anchor is None:
            return 0
            
        anchor_num, n_view = X_anchor.shape[0], X_anchor.shape[1]


        y_anchor = y_anchor.contiguous().view(-1, 1)

        anchor_count = n_view
        # X_anchor:(3 dim) total_classes x n_view x feat_dim 
        # anchor_feature:(2 dim) (total_classes x n_view) x feat_dim
        anchor_feature = torch.cat(torch.unbind(X_anchor, dim=1), dim=0)

        if queue is not None:
            X_contrast, y_contrast = self._sample_negative(queue)
            y_contrast = y_contrast.contiguous().view(-1, 1)
            contrast_count = 1
            contrast_feature = X_contrast
            # y_contrast = y_anchor
            # contrast_count = 1
            # contrast_feature = queue
        else:
            y_contrast = y_anchor
            contrast_count = n_view
            # contrast_feature = torch.cat(torch.unbind(X_anchor, dim=1), dim=0)
            contrast_feature = anchor_feature

        # mask: total_classes x num_of_class
        mask = torch.eq(y_anchor, y_contrast.T).float().cuda()

        # anchor_dot_contrast : n x num_of_class
        anchor_dot_contrast = torch.div(torch.matmul(anchor_feature, contrast_feature.T),
                                        self.temperature)
        logits_max, _ = torch.max(anchor_dot_contrast, dim=1, keepdim=True)
        # logits : n x num_of_class
        logits = anchor_dot_contrast - logits_max.detach()

        mask = mask.repeat(anchor_count, contrast_count)
        
        neg_mask = 1 - mask

        ## 用于回避自己作为自己的正例，但使用memory bank的情况可以不用
        # mask = mask * logits_mask
        # # 去掉对角线
        # logits_mask = torch.ones_like(mask).scatter_(1,
        #                                              torch.arange(anchor_num * anchor_count).view(-1, 1).cuda(),
        #                                              0)
        
        
        # mask = mask * logits_mask

        neg_logits = torch.exp(logits) * neg_mask
        neg_logits = neg_logits.sum(1, keepdim=True)

        exp_logits = torch.exp(logits)

        log_prob = logits - torch.log(exp_logits + neg_logits)

        mean_log_prob_pos = (mask * log_prob).sum(1) / mask.sum(1)

        loss = - (self.temperature / self.base_temperature) * mean_log_prob_pos
        loss = loss.mean()

        return loss

    def forward(self, feats, labels=None, predict=None, queue=None):
        labels = labels.unsqueeze(1).float().clone()
        
        labels = torch.nn.functional.interpolate(labels,
                                                 (feats.shape[2], feats.shape[3]), mode='nearest')
        labels = labels.squeeze(1).long()
        assert labels.shape[-1] == feats.shape[-1], '{} {}'.format(labels.shape, feats.shape)

        batch_size = feats.shape[0]

        labels = labels.contiguous().view(batch_size, -1)
        predict = predict.contiguous().view(batch_size, -1)
        feats = feats.permute(0, 2, 3, 1)
        feats = feats.contiguous().view(feats.shape[0], -1, feats.shape[-1])
    
        feats_, labels_ = self._hard_anchor_sampling(feats, labels, predict)

        loss = self._contrastive(feats_, labels_, queue=queue)
        return loss

    

class ContrastCELoss(nn.Module, ABC):
    def __init__(self, configer=None):
        super(ContrastCELoss, self).__init__()

        self.configer = configer

        ignore_index = -1
        if self.configer.exists('loss', 'params') and 'ce_ignore_index' in self.configer.get('loss', 'params'):
            ignore_index = self.configer.get('loss', 'params')['ce_ignore_index']
        Log.info('ignore_index: {}'.format(ignore_index))

        self.loss_weight = self.configer.get('contrast', 'loss_weight')
        self.use_rmi = self.configer.get('contrast', 'use_rmi')
        self.use_lovasz = self.configer.get('contrast', 'use_lovasz')

        if self.use_rmi:
            self.seg_criterion = FSRMILoss(configer=configer)
        elif self.use_lovasz:
            self.seg_criterion = FSCELOVASZLoss(configer=configer)
        else:
            self.seg_criterion = FSCELoss(configer=configer)

        self.contrast_criterion = PixelContrastLoss(configer=configer)

    def forward(self, preds, target, with_embed=False):
        h, w = target.size(1), target.size(2)

        assert "seg" in preds
        assert "embed" in preds

        seg = preds['seg']
        embedding = preds['embed']

        if "segment_queue" in preds:
            segment_queue = preds['segment_queue']
        else:
            segment_queue = None

        if "pixel_queue" in preds:
            pixel_queue = preds['pixel_queue']
        else:
            pixel_queue = None

        pred = F.interpolate(input=seg, size=(h, w), mode='bilinear', align_corners=True)
        loss = self.seg_criterion(pred, target)

        if segment_queue is not None and pixel_queue is not None:
            queue = torch.cat((segment_queue, pixel_queue), dim=1)

            _, predict = torch.max(seg, 1)
            loss_contrast = self.contrast_criterion(embedding, target, predict, queue)
        else:
            loss_contrast = 0

        if with_embed is True:
            return loss + self.loss_weight * loss_contrast

        return loss + 0 * loss_contrast  # just a trick to avoid errors in distributed training


class ContrastAuxCELoss(nn.Module, ABC):
    def __init__(self, configer=None):
        super(ContrastAuxCELoss, self).__init__()

        self.configer = configer

        ignore_index = -1
        if self.configer.exists('loss', 'params') and 'ce_ignore_index' in self.configer.get('loss', 'params'):
            ignore_index = self.configer.get('loss', 'params')['ce_ignore_index']
        Log.info('ignore_index: {}'.format(ignore_index))

        self.loss_weight = self.configer.get('contrast', 'loss_weight')
        self.use_rmi = self.configer.get('contrast', 'use_rmi')

        if self.use_rmi:
            self.seg_criterion = FSAuxRMILoss(configer=configer)
        else:
            self.seg_criterion = FSAuxCELoss(configer=configer)

        self.contrast_criterion = PixelContrastLoss(configer=configer)

    def forward(self, preds, target):
        h, w = target.size(1), target.size(2)

        assert "seg" in preds
        assert "seg_aux" in preds

        seg = preds['seg']
        seg_aux = preds['seg_aux']

        embedding = preds['embedding'] if 'embedding' in preds else None

        pred = F.interpolate(input=seg, size=(h, w), mode='bilinear', align_corners=True)
        pred_aux = F.interpolate(input=seg_aux, size=(h, w), mode='bilinear', align_corners=True)
        loss = self.seg_criterion([pred_aux, pred], target)

        if embedding is not None:
            _, predict = torch.max(seg, 1)

            loss_contrast = self.contrast_criterion(embedding, target, predict)
            return loss + self.loss_weight * loss_contrast

        return loss


class PixelPrototypeDistanceLoss(nn.Module):
    def __init__(self, configer=None):
        super(PixelPrototypeDistanceLoss, self).__init__()
        
        self.configer = configer
        self.ignore_index = self.configer.get('loss', 'ignore_index')
    
    def forward(self, emb, lb, segment_queue):
        shapedEmb = emb.permute(0,2,3,1)[lb!=self.ignore_index,:]
        # shapedEmb = shapedEmb.contiguous().view(-1, emb.shape[1])
        simScore = torch.matmul(shapedEmb, segment_queue.T)
        # simScore = torch.einsum('bc,nc->bn', shapedEmb, segment_queue)
        # print(simScore)
        # simScoreTrue = simScore[lb!=self.ignore_index,:]
        # print(simScore)
        useful_lb = lb[lb!=self.ignore_index]
        # print(useful_lb)
        logits = torch.gather(simScore, 1, useful_lb[:, None].long())
        # print(logits)
        loss_ppd = (1 - logits).pow(2).mean()

        return loss_ppd

class PixelContrastLossOnlyNeg(nn.Module, ABC):
    def __init__(self, configer):
        super(PixelContrastLossOnlyNeg, self).__init__()

        self.configer = configer
        self.temperature = self.configer.get('contrast', 'temperature')
        self.base_temperature = self.configer.get('contrast', 'base_temperature')

        self.max_samples = self.configer.get('contrast', 'max_samples')
        self.max_views = self.configer.get('contrast', 'max_views')
        self.num_unify_classes = self.configer.get('num_unify_classes')


    def forward(self, feats, labels=None, queue=None):
        # labels: batch_size x h x w x num_of_class
        # 1表示目标类，0表示非目标类
        # batch_size = feats.shape[0]
        b, c, h, w = feats.shape

        feats = feats.permute(0, 2, 3, 1)
        feats = feats.contiguous().view(-1, feats.shape[-1])

        anchor_dot_contrast = torch.div(torch.matmul(feats, queue.clone().T), self.temperature)
        
        pos_mask = labels.detach().contiguous().view(-1, self.num_unify_classes)
        # neg_mask = 1 - pos_mask
        neg_mask = pos_mask.logical_not()
        
        neg_logits = torch.exp(anchor_dot_contrast) * neg_mask
        pos_logits = torch.exp(-anchor_dot_contrast) * pos_mask
        
        sum_logits = torch.sum(neg_logits, dim=1) * torch.sum(pos_logits, dim=1) + 1
        
        # print("self.temperature: ", self.temperature)
        # print("feats: ", torch.isnan(feats).any())
        # print("queue: ", torch.isnan(queue).any())
        # print("anchor_dot_contrast: ", torch.isnan(anchor_dot_contrast).any())
        # print("neg_logits: ", torch.isnan(neg_logits).any())
        # print("pos_logits: ", torch.isnan(pos_logits).any())
        # print("sum_logits: ", torch.isnan(sum_logits).any())
        loss = torch.log(sum_logits)
        
        lb_num = torch.sum(loss != 0)
        if lb_num ==0:
            return 0
        
        loss = torch.sum(loss) / lb_num 

        return loss

class PixelContrastLossMulProto(nn.Module, ABC):
    def __init__(self, configer):
        super(PixelContrastLossMulProto, self).__init__()

        self.configer = configer
        self.temperature = self.configer.get('contrast', 'temperature')
        
        self.num_unify_classes = self.configer.get('num_unify_classes')
        
        self.soft_plus = nn.Softplus()


    def forward(self, feats, labels=None):
        # labels: batch_size x h x w x n
        # 1表示目标类，0表示非目标类
        # batch_size = feats.shape[0]
        b, n = feats.shape

        # feats = feats.contiguous().view(-1, n)

        anchor_dot_contrast = torch.div(feats, self.temperature)
        
        pos_mask = labels.detach().contiguous().view(-1, n).bool()
        # neg_mask = 1 - pos_mask
        neg_mask = pos_mask.logical_not()
        
        pos_pred = -anchor_dot_contrast
        neg_pred = anchor_dot_contrast
        # pos_pred = -pred * self.gamma #* pos_lb
        # neg_pred = (pred + self.m) * self.gamma #* neg_lb
        pos_pred[neg_mask] = -1e12
        neg_pred[pos_mask] = -1e12
        loss = self.soft_plus(torch.logsumexp(neg_pred, dim=1) + torch.logsumexp(pos_pred, dim=1))
        
        # neg_logits = torch.exp(anchor_dot_contrast) * neg_mask
        # pos_logits = torch.exp(-anchor_dot_contrast) * pos_mask
        
        # sum_logits = torch.sum(neg_logits, dim=1) * torch.sum(pos_logits, dim=1) + 1
    
        # loss = torch.log(sum_logits)
        
        lb_num = torch.sum(loss != 0)
        if lb_num ==0:
            return 0
        
        loss = torch.sum(loss) / lb_num 

        return loss

def test_PixelContrastLossMulProto():

    
    feat = torch.randn(3,16,4,4)
    queue = torch.randn(4,16)
    lb = torch.rand(3,4,4,4)
    lb[lb < 0.5] = 0
    lb[lb >= 0.5] = 1
    lb = lb.bool()
    print(lb)
    
    ft = rearrange(feat, 'b c h w -> (b h w) c')
    logit = torch.mm(ft, queue.T)
    configer = Configer(configs='configs/test.json')
    loss1 = PixelContrastLossMulProto(configer)
    loss2 = PixelContrastLossOnlyNeg(configer)
    print(loss1(logit, lb))
    print(loss2(feat, lb, queue))
    
    

if __name__ == "__main__":
    from einops import rearrange
    from tools.configer import *
    test_PixelContrastLossMulProto()