import torch
import torch.nn as nn
import torch.nn.functional as F


class ClassRemap():
    def __init__(self, configer=None):
        self.configer = configer
        self.ignore_index = self.configer.get('loss', 'ignore_index')
        self.temperature = self.configer.get('contrast', 'temperature')
        self.remapList = []
        self.maxMapNums = [] # 最大映射类别数
        # self.class_weight = self.configer.get("class_weight")
        self.num_unify_classes = self.configer.get('num_unify_classes')
        self.softmax = nn.Softmax(dim=1)
        self.network_stride = self.configer.get('network', 'stride')
        self.num_prototype = self.configer.get('contrast', 'num_prototype')
        self.Upsample = nn.Upsample(scale_factor=self.network_stride, mode='nearest')
        self.max_iter = self.configer.get('lr', 'max_iter')
        self._unpack()
        
    def SingleSegRemapping(self, labels, dataset_id):
        ## 只输出 唯一映射部分
        ## dataset_id指定映射方案
        outLabels = []
        
        mask = torch.ones_like(labels) * self.ignore_index
        
        for k, v in self.remapList[dataset_id].items():
            if len(v) > 1: 
                continue

            mask[labels==int(k)] = v[0]

            
        return mask    
        
    def SegRemapping(self, labels, dataset_id):
        ## 输入 batchsize x H x W 输出 k x batch size x H x W
        ## dataset_id指定映射方案
        outLabels = []
        
        for i in range(0, self.maxMapNums[dataset_id]):
            mask = torch.ones_like(labels) * self.ignore_index
            
            for k, v in self.remapList[dataset_id].items():
                if len(v) <= i: 
                    continue

                mask[labels==int(k)] = v[i]
                
            outLabels.append(mask)
            
        return outLabels
        
    
    def ContrastRemapping(self, labels, embed, proto, dataset_id):
        is_emb_upsampled = self.configer.get('contrast', 'upsample')
        
        # if is_emb_upsampled:
        #     contrast_lb = labels[:, ::self.network_stride, ::self.network_stride]
        # else:
        contrast_lb = labels
            
        # contrast_lb = labels
        B, H, W = contrast_lb.shape
        
        mask = torch.ones_like(contrast_lb) * self.ignore_index
        weight_mask = torch.zeros([B, H, W, self.num_unify_classes], dtype=torch.float32)
        if embed.is_cuda:
            weight_mask = weight_mask.cuda()
        
        for k, v in self.remapList[dataset_id].items():
            # 判断是否为空
            if not (contrast_lb==int(k)).any():
                continue
            
            
            if len(v) == 1: 
                mask[contrast_lb==int(k)] = v[0]
                
                weight_vector = torch.zeros(self.num_unify_classes, dtype=torch.float32)
                if weight_mask.is_cuda:
                    weight_vector = weight_vector.cuda()
                    
                weight_vector[v[0]] = 1
                weight_mask[contrast_lb==int(k)] = weight_vector
            else:
                # 在多映射情况下，找到内积最大的一项的标签
                
                shapeEmbed = embed.permute(0,2,3,1).detach()
                # shapeEmbed = F.normalize(shapeEmbed, p=2, dim=-1)
                # print(shapeEmbed[labels==int(k)].shape)
                # print(proto.shape)
                # print(proto[v].shape)
                # n: b x h x w, d: dim, c:  num of class
                simScore = torch.div(torch.einsum('nd,cd->nc', shapeEmbed[contrast_lb==int(k)], proto[v]), self.temperature)
                MaxSimIndex = torch.max(simScore, dim=1)[1]
                if mask.is_cuda:
                    mask[contrast_lb==int(k)] = torch.tensor(v, dtype=torch.bool)[MaxSimIndex].cuda()
                else:
                    mask[contrast_lb==int(k)] = torch.tensor(v, dtype=torch.bool)[MaxSimIndex]
                
                expend_vector = torch.zeros([simScore.shape[0], self.num_unify_classes], dtype=torch.float32)
                if weight_mask.is_cuda:
                    expend_vector = expend_vector.cuda()
                    
                
                expend_vector[:, v] = self.softmax(simScore)
                
                weight_mask[contrast_lb==int(k)] = expend_vector
                
        # if is_emb_upsampled:
        #     weight_mask = self.Upsample(weight_mask.permute(0,3,1,2)).permute(0,2,3,1)
          
        return mask, weight_mask
        
    def GetEqWeightMask(self, labels, dataset_id):
        B, H, W = labels.shape
        weight_mask = torch.zeros([B, H, W, self.num_unify_classes], dtype=torch.float32)
        if labels.is_cuda:
            weight_mask = weight_mask.cuda()
            
        for k, v in self.remapList[dataset_id].items():
            map_num = len(v)
            weight_vector = torch.zeros(self.num_unify_classes, dtype=torch.float32)
            if weight_mask.is_cuda:
                weight_vector = weight_vector.cuda()
                
            for val in v:
                weight_vector[val] = 1  # / map_num
                
            weight_mask[labels==int(k)] = weight_vector

        
        return weight_mask
    
    
    def _unpack(self):
        self.n_datasets = -1
        if self.configer.exists('n_datasets'):
            self.n_datasets = self.configer.get('n_datasets')
        else:
            raise NotImplementedError("read json errror! no  n_datasets")    

        self.class_weight = []
        for i in range(0, self.num_unify_classes):
            self.class_weight.append(self.configer.get('class_weight')[str(i)])
            
        self.class_weight = torch.tensor(self.class_weight)

        # 读取class remap info
        for i in range(1, self.n_datasets+1):
            if not self.configer.exists('class_remap' + str(i)):
                raise  NotImplementedError("read json errror! no class_remap"+str(i))
            class_id = 0
            maxMapNum = 0
            class_remap = {}
            while str(class_id) in self.configer.get('class_remap' + str(i)):
                class_remap[class_id] = self.configer.get('class_remap' + str(i))[str(class_id)]
                maxMapNum = len(class_remap[class_id]) if len(class_remap[class_id]) > maxMapNum else maxMapNum
                class_id += 1
            self.remapList.append(class_remap)
            self.maxMapNums.append(maxMapNum)
            
    def getAnyClassRemap(self, lb_id, dataset_id):
        return self.remapList[dataset_id][lb_id]
    
    def ReverseSegRemap(self, preds, dataset_id):
        # logits : B x h x w
        Remap_pred = torch.ones_like(preds) * self.ignore_index
        for k, v in self.remapList[dataset_id].items():
            for lb in v:
                Remap_pred[preds == int(lb)] = int(k)
                
        return Remap_pred
    
    def ExpendRemapByPrototypeNum(self, v):
        
        outV = torch.Tensor()
        for i in v:

            outV = torch.cat((outV, torch.arange(i*self.num_prototype, (i+1)*self.num_prototype))).long()
            
        return outV
    
    def get_class_weight(self,cur_class_id,dataset_id):
        tgt_classes = self.remapList[dataset_id][cur_class_id]        
        # weight_vec = torch.ones(len(tgt_classes), dtype=torch.float)
        weight_vec = self.class_weight[tgt_classes]
        return weight_vec
        
    
        
class ClassRemapOneHotLabel(ClassRemap):
    def __init__(self, configer=None):
        super(ClassRemapOneHotLabel, self).__init__(configer)
        self.update_sim_thresh = self.configer.get('contrast', 'update_sim_thresh')

    def SegRemapping(self, labels, dataset_id):
        ## 输入 batchsize x H x W 输出 n x batch size x H x W
        ## 1表示目标类， 0表示非目标类
        ## dataset_id指定映射方案
        b, h, w = labels.shape
        outMultiLabels = torch.zeros([b, h, w, self.num_unify_classes],dtype=torch.bool)
        if labels.is_cuda:
            outMultiLabels = outMultiLabels.cuda()
        
        for k, v in self.remapList[dataset_id].items():

            if len(v) == 1:
                outMultiLabels[labels==int(k), v] = 1
            else:
                outMultiLabels[labels==int(k), torch.tensor(v).unsqueeze(1)] = 1

        return outMultiLabels
    
    def ContrastRemapping(self, labels, embed, proto, dataset_id):
        # seg_mask 监督分割头的多类别标签
        # contrast_mask 监督投影头的标签信息
        is_emb_upsampled = self.configer.get('contrast', 'upsample')
        if is_emb_upsampled:
            raise Exception("Not Imp")
        
        contrast_lb = labels[:, ::self.network_stride, ::self.network_stride]

        B, H, W = labels.shape
        _, h_c, w_c = contrast_lb.shape
        # contrast_mask = torch.ones_like(contrast_lb) * self.ignore_index
        
        ## sig_seg_mask : 单标签矩阵
        ## Seg_temp_mask : 在投影头监督下的单标签矩阵，保存置信度高的标签
        ## hard_lb_mask : 在投影头监督下的单标签矩阵，保存置信度低的标签
        
        # Seg_temp_mask = torch.zeros([B, h_c, w_c, self.num_unify_classes], dtype=torch.bool)
        contrast_lb_mask = torch.zeros([B, h_c, w_c, self.num_unify_classes], dtype=torch.bool)
        if labels.is_cuda:
            # sig_seg_mask = sig_seg_mask.cuda()
            # Seg_temp_mask = Seg_temp_mask.cuda()
            contrast_lb_mask = contrast_lb_mask.cuda()
        
        ## 循环两遍，单标签覆盖多标签
        for k, v in self.remapList[dataset_id].items():
            # 判断是否为空
            if not (labels==int(k)).any():
                continue
            
            if len(v) != 1: 
                # 在多映射情况下，找到内积最大的一项的标签
                if not (contrast_lb==int(k)).any():
                    continue
                
                shapeEmbed = embed.permute(0,2,3,1).detach()

                # # n: b x h x w, d: dim, c:  num of class
                # weight_vec = self.get_class_weight(k, dataset_id)
                # if proto.is_cuda:
                #     weight_vec = weight_vec.cuda()
                
                
                # weighted_proto = torch.einsum('c,cd->cd', weight_vec, proto)
                simScore = torch.einsum('nd,cd->nc', shapeEmbed[contrast_lb==int(k)], proto)

                
                # simScore = self.softmax(simScore)
                
                MaxSim, MaxSimIndex = torch.max(simScore, dim=1)
                outputIndex = torch.ones_like(MaxSimIndex) * self.ignore_index
                
                # 新标签样本
                RemapLbIndex = torch.zeros([MaxSimIndex.shape[0], self.num_unify_classes], dtype=torch.bool)
                # 筛选出目标类的index。对于最大值不在目标类的情况以及最大值没超过阈值的pixel，标签设置为忽略

                hardLd = torch.zeros(self.num_unify_classes, dtype=torch.bool)
                tensor_v = torch.tensor(v)
                if labels.is_cuda:
                    tensor_v = tensor_v.cuda()
                    RemapLbIndex = RemapLbIndex.cuda()
                    hardLd = hardLd.cuda()
                
                targetIndex = MaxSimIndex.unsqueeze(1).eq(tensor_v)
                targetIndex = torch.unique(targetIndex.nonzero(as_tuple=True)[0])
                MaxSimIndex[MaxSim < self.update_sim_thresh] = self.ignore_index
                
                outputIndex[targetIndex] = MaxSimIndex[targetIndex]
                cur_iter = self.configer.get('iter')
                for class_id in v:
                    this_classes = MaxSim[outputIndex==class_id]
                    len_this_classes = this_classes.shape[0]
                    if len_this_classes == 0:
                        continue
                    else:
                        ratio = min(1.25 * float(cur_iter) / self.max_iter, 1)
                        len_this_classes = max(int(len_this_classes * ratio), 1)
                        
                    out_vector = torch.ones_like(this_classes) * self.ignore_index
                    
                    
                    topkindex = torch.topk(this_classes, len_this_classes, sorted=False)[1]
                    out_vector[topkindex] = class_id
                    outputIndex[outputIndex==class_id] = out_vector.long()
                
                
                hardLd[tensor_v] = 1
               
                
                RemapLbIndex[outputIndex!=self.ignore_index, outputIndex[outputIndex!=self.ignore_index]] = 1
   
                # contrast_mask[contrast_lb==int(k)] = outputIndex
                RemapLbIndex[outputIndex==self.ignore_index] = hardLd
                contrast_lb_mask[contrast_lb==int(k)] = RemapLbIndex
    
                


        seg_mask = contrast_lb_mask.permute(0, 3, 1, 2)
        seg_mask = F.interpolate(seg_mask.float(), size=(H,W), mode='nearest').bool()
        seg_mask = seg_mask.permute(0, 2, 3, 1)

        zero_vector = torch.zeros(self.num_unify_classes, dtype=torch.bool)
        if seg_mask.is_cuda:
            zero_vector = zero_vector.cuda()
        
        for k, v in self.remapList[dataset_id].items():
            # 判断是否为空
            if not (labels==int(k)).any():
                continue
            
            
            if len(v) == 1: 
                contrast_lb_mask[contrast_lb==int(k), v] = 1
                
                seg_vector = torch.zeros(self.num_unify_classes, dtype=torch.bool)
                if seg_mask.is_cuda:
                    seg_vector = seg_vector.cuda()
                    
                seg_vector[v[0]] = 1

                seg_mask[labels==int(k)] = seg_vector
            else:
                expend_vector = torch.zeros(self.num_unify_classes, dtype=torch.bool)
                if seg_mask.is_cuda:
                    expend_vector = expend_vector.cuda()

                expend_vector[v] = 1
                # 寻找为0项
                lb_k_mask = seg_mask[labels==int(k)]
                # maxValue = torch.max(lb_k_mask, dim=1)[0]
                # lb_k_mask[maxValue==0] = expend_vector
                
                sumValue = torch.sum(lb_k_mask, dim=1)
                lb_k_mask[sumValue==0] = expend_vector
                # lb_k_mask[sumValue==1] = zero_vector
                seg_mask[labels==int(k)] = lb_k_mask

        seg_mask[labels==self.ignore_index] = zero_vector
        # seg_mask[seg_mask==self.ignore_index] = 0
        
        # if is_emb_upsampled:
        #     weight_mask = self.Upsample(weight_mask.permute(0,3,1,2)).permute(0,2,3,1)
          
        return contrast_lb_mask, seg_mask
  
    
    def MultiProtoRemapping(self, labels, proto_logits, dataset_id):
        contrast_lb = labels[:, ::self.network_stride, ::self.network_stride]
        
        B, H, W = labels.shape
        _, h_c, w_c = contrast_lb.shape
        
        ## sig_seg_mask : 单标签矩阵
        ## Seg_temp_mask : 在投影头监督下的单标签矩阵，保存置信度高的标签
        ## hard_lb_mask : 在投影头监督下的单标签矩阵，保存置信度低的标签
        
        # Seg_temp_mask = torch.zeros([B, h_c, w_c, self.num_unify_classes], dtype=torch.bool)
        contrast_lb_mask = torch.zeros([B, h_c, w_c, self.num_unify_classes*self.num_prototype], dtype=torch.bool)
        seg_mask = torch.zeros([B, h_c, w_c, self.num_unify_classes], dtype=torch.bool)
        if labels.is_cuda:
            contrast_lb_mask = contrast_lb_mask.cuda()
            seg_mask = seg_mask.cuda()
        
        # simScore :  (b h_c w_c) * (n k)
        # simScore = torch.div(proto_logits.detach(), self.temperature)
        # simScore = self.softmax(simScore)
        simScore = proto_logits
        OneColContrastLb = contrast_lb.contiguous().view(-1)
        
        ## 循环两遍，单标签覆盖多标签
        for k, v in self.remapList[dataset_id].items():
            # 判断是否为空
            if not (labels==int(k)).any():
                continue
            
            if len(v) != 1: 
                # 在多映射情况下，找到内积最大的一项的标签
                if not (contrast_lb==int(k)).any():
                    continue
                       
                MaxSim, MaxSimIndex = torch.max(simScore[OneColContrastLb == k], dim=1)
                # print(MaxSim)
                outputIndex = torch.ones_like(MaxSimIndex) * self.ignore_index
                
                # 新标签样本
                RemapLbIndex = torch.zeros([MaxSimIndex.shape[0], self.num_unify_classes*self.num_prototype], dtype=torch.bool)
                # 筛选出目标类的index。对于最大值不在目标类的情况以及最大值没超过阈值的pixel，标签设置为忽略

                hardLd = torch.zeros(self.num_unify_classes*self.num_prototype, dtype=torch.bool)
                Expendtensor_v = self.ExpendRemapByPrototypeNum(v)
                if labels.is_cuda:
                    Expendtensor_v = Expendtensor_v.cuda()
                    RemapLbIndex = RemapLbIndex.cuda()
                    hardLd = hardLd.cuda()
                
                targetIndex = MaxSimIndex.unsqueeze(1).eq(Expendtensor_v)
                targetIndex = torch.unique(targetIndex.nonzero(as_tuple=True)[0])
                MaxSimIndex[MaxSim < self.update_sim_thresh] = self.ignore_index
                
                outputIndex[targetIndex] = MaxSimIndex[targetIndex]
                cur_iter = self.configer.get('iter')
                
                for class_id in Expendtensor_v:
                    this_classes = MaxSim[outputIndex==class_id]
                    len_this_classes = this_classes.shape[0]
                    if len_this_classes == 0:
                        continue
                    else:
                        ratio = min(1.25 * float(cur_iter) / self.max_iter, 1)
                        len_this_classes = max(int(len_this_classes * ratio), 1)
                        
                    out_vector = torch.ones_like(this_classes, dtype=torch.long) * self.ignore_index
                    
                    
                    topkindex = torch.topk(this_classes, len_this_classes, sorted=False)[1]
                    out_vector[topkindex] = class_id
                    outputIndex[outputIndex==class_id] = out_vector.long()

                hardLd[Expendtensor_v] = 1
               
                RemapLbIndex[outputIndex==self.ignore_index] = hardLd
                RemapLbIndex[outputIndex!=self.ignore_index, outputIndex[outputIndex!=self.ignore_index]] = 1
                
                # contrast_mask[contrast_lb==int(k)] = outputIndex
                contrast_lb_mask[contrast_lb==int(k)] = RemapLbIndex
        
        for i in range(0, self.num_prototype):
            seg_mask += contrast_lb_mask[:,:,:,i::self.num_prototype]

        seg_mask = seg_mask.permute(0, 3, 1, 2)
        seg_mask = F.interpolate(seg_mask.float(), size=(H,W), mode='nearest').bool()
        seg_mask = seg_mask.permute(0, 2, 3, 1)

        zero_vector = torch.zeros(self.num_unify_classes, dtype=torch.bool)
        if seg_mask.is_cuda:
            zero_vector = zero_vector.cuda()
        
        for k, v in self.remapList[dataset_id].items():
            # 判断是否为空
            if not (labels==int(k)).any():
                continue
            
            
            if len(v) == 1: 
                MaxSim, MaxSimIndex = torch.max(simScore[OneColContrastLb == k, v[0]*self.num_prototype:(v[0]+1)*self.num_prototype], dim=1)
                contrast_lb_mask[contrast_lb==int(k), v[0]*self.num_prototype + MaxSimIndex] = 1
                
                seg_vector = torch.zeros(self.num_unify_classes, dtype=torch.bool)
                if seg_mask.is_cuda:
                    seg_vector = seg_vector.cuda()
                    
                seg_vector[v[0]] = 1

                seg_mask[labels==int(k)] = seg_vector
                
                # if int(k) == 3:
                #     print(dataset_id)
                #     print(seg_vector)
            else:
                expend_vector = torch.zeros(self.num_unify_classes, dtype=torch.bool)
                if seg_mask.is_cuda:
                    expend_vector = expend_vector.cuda()

                expend_vector[v] = 1
                # 寻找为0项
                lb_k_mask = seg_mask[labels==int(k)]
                # maxValue = torch.max(lb_k_mask, dim=1)[0]
                # lb_k_mask[maxValue==0] = expend_vector
                
                sumValue = torch.sum(lb_k_mask, dim=1)
                lb_k_mask[sumValue==0] = expend_vector
                # lb_k_mask[sumValue==1] = zero_vector
                seg_mask[labels==int(k)] = lb_k_mask

        seg_mask[labels==self.ignore_index] = zero_vector
        # seg_mask[seg_mask==self.ignore_index] = 0
        
        # if is_emb_upsampled:
        #     weight_mask = self.Upsample(weight_mask.permute(0,3,1,2)).permute(0,2,3,1)
          
        return contrast_lb_mask, seg_mask
    
    def KMeansRemapping(self, labels, dataset_id):
        ## 只输出 唯一映射部分
        ## dataset_id指定映射方案
        outLabels = []
        
        cluster_mask = torch.zeros_like(labels).bool() 
        constraint_mask = torch.zeros((*(labels.shape), self.num_unify_classes), dtype=torch.bool)
        if labels.is_cuda:
            constraint_mask = constraint_mask.cuda()
        
        for k, v in self.remapList[dataset_id].items():
            if len(v) > 1: 
                expend_vector = torch.zeros(self.num_unify_classes, dtype=torch.bool)
                if labels.is_cuda:
                    expend_vector = expend_vector.cuda()

                expend_vector[v] = True
                cluster_mask[labels==int(k)] = True
                constraint_mask[labels==int(k)] = expend_vector         
            
        return cluster_mask, constraint_mask
    
    def UpsampleProtoTarget(self, labels, proto_target, dataset_id):
        
        B, H, W = labels.shape
        
        # proto_target = proto_target.permute(0, 3, 1, 2)
        seg_mask = F.interpolate(proto_target.unsqueeze(1).float(), size=(H,W), mode='nearest').squeeze(1).long()
        # seg_mask = seg_mask.permute(0, 2, 3, 1)

        zero_vector = torch.zeros(self.num_unify_classes, dtype=torch.bool)
        if seg_mask.is_cuda:
            zero_vector = zero_vector.cuda()
        
        for k, v in self.remapList[dataset_id].items():
            # 判断是否为空
            if not (labels==int(k)).any():
                continue
            
            
            if len(v) == 1: 
                seg_mask[labels==int(k)] = v[0]
                
                # if int(k) == 3:
                #     print(dataset_id)
                #     print(seg_vector)
            else:

                seg_mask[labels==int(k)] = self.ignore_index

        seg_mask[labels==self.ignore_index] = self.ignore_index
        # seg_mask[seg_mask==self.ignore_index] = 0
        
        # if is_emb_upsampled:
        #     weight_mask = self.Upsample(weight_mask.permute(0,3,1,2)).permute(0,2,3,1)
          
        return seg_mask
    
def test_ContrastRemapping():
    configer = Configer(configs='configs/test.json')
    classRemap = ClassRemapOneHotLabel(configer)
    labels = torch.tensor([[2, 0, 0, 0],
                           [2, 1, 1, 1],
                           [2, 2, 1, 2],
                           [0, 0, 0, 2]]).unsqueeze(0)
    segment_queue = torch.tensor([[-1, 0], [1, 0], [0, 1], [0, -1]], dtype=torch.float) # 19 x 2
    embed = torch.tensor([[[-0.1, 0.9],[0.9, 0.1]],
                          [[-0.8, 0.2],[-0.1, 0.9]]]).unsqueeze(0)
    # 
    # embed = embed.permute(0, 3, 1, 2)
    Shapedembed = embed.contiguous().view(-1, 2)
    mul_segment_queue = torch.tensor([[[-1, 0], [0.9, 0.1], [-0.1, 1], [0, -1]],
                                      [[-0.9, 0.1], [1, 0], [0, 1], [-0.1, -1.9]]], dtype=torch.float) # 19 x 2 x 2
    proto_logits = torch.mm(Shapedembed, mul_segment_queue.view(-1, 2).T)
    print(mul_segment_queue.view(-1, 2))
    # print(proto_logits)
    rearrange_logit = torch.zeros_like(proto_logits)

    for i in range(0, 2):
        rearrange_logit[:, i::2] = proto_logits[:, i*4:(i+1)*4]
    print(rearrange_logit)
    
    # contrast_mask, seg_mask = classRemap.ContrastRemapping(labels, embed, segment_queue, 0)
    contrast_mask, seg_mask = classRemap.MultiProtoRemapping(labels, rearrange_logit, 0)
    print(contrast_mask)
    print(seg_mask) 
    

        
if __name__ == "__main__":
    import sys
    sys.path.insert(0, '/home/cxh/mr/BiSeNet')
    from tools.configer import *
    from math import sin, cos
    from einops import rearrange
    
    test_ContrastRemapping()
    # pi = 3.14

    # configer = Configer(configs='configs/bisenetv2_city_cam.json')
    # classRemap = ClassRemap(configer=configer)
    # labels = torch.tensor([[[0, 6, 0], [1, 0, 1]]], dtype=torch.float) # 1 x 2 x 3
    # embed = torch.tensor([[[[cos(pi/4), cos(pi/3), cos(2*pi/3)], [cos(7*pi/6), cos(3*pi/2), cos(5*pi/3)]], 
    #                        [[sin(pi/4), sin(pi/3), sin(2*pi/3)], [sin(7*pi/6), sin(3*pi/2), sin(5*pi/3)]]]], requires_grad=True) # 1 x 1 x 2 x 3
    # proto = torch.tensor([[-1, 0], [1, 0], [0, 1], [0, -1]], dtype=torch.float) # 19 x 2
    # # print(classRemap.ContrastRemapping(labels, embed, proto, 2))
    # print(classRemap.GetEqWeightMask(labels, 1))
    # # print(classRemap.getAnyClassRemap(3, 0))
    

    