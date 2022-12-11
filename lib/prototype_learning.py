import sys
sys.path.insert(0, 'D:/Study/code/BiSeNet')

from lib.sinkhorn import distributed_sinkhorn
from lib.momentum_update import momentum_update
from timm.models.layers import trunc_normal_
from lib.module.kmeans import kmeans
from einops import rearrange, repeat

import torch
import torch.nn as nn
import torch.nn.functional as F

def prototype_learning(configer, prototypes, _c, out_seg, gt_seg, update_prototype=False):
    
    num_unify_classes = configer.get('num_unify_classes')
    num_prototype = configer.get('contrast', 'num_prototype')
    coefficient = configer.get('contrast', 'coefficient')
    network_stride = configer.get('network', 'stride')
    ## pred_seg 分割头预测的类别
    ## gt_seg: (bhw) 的一维向量
    ## mask: 预测正确组
    pred_seg = torch.max(out_seg[:, :, ::network_stride, ::network_stride], 1)[1]
    mask = (gt_seg == pred_seg.view(-1))

    # n: b*h*w, k: num_class, m: num_prototype
    masks = torch.einsum('nd,kmd->nmk', _c, prototypes)
    cosine_similarity = torch.mm(_c, prototypes.view(-1, prototypes.shape[-1]).t())

    proto_logits = cosine_similarity
    proto_target = gt_seg.clone().long()

    # clustering for each class
    protos = prototypes.data.clone()
    for k in range(num_unify_classes):
        init_q = masks[..., k]
        init_q = init_q[gt_seg == k, ...]
        if init_q.shape[0] == 0:
            continue

        q, indexs = distributed_sinkhorn(init_q)

        m_k = mask[gt_seg == k]

        c_k = _c[gt_seg == k, ...]

        m_k_tile = repeat(m_k, 'n -> n tile', tile=num_prototype)
        # 只用预测分割头正确的特征进行更新
        m_q = q * m_k_tile  # n x self.num_prototype

        c_k_tile = repeat(m_k, 'n -> n tile', tile=c_k.shape[-1])

        c_q = c_k * c_k_tile  # n x embedding_dim

        f = m_q.transpose(0, 1) @ c_q  # self.num_prototype x embedding_dim

        n = torch.sum(m_q, dim=0)

        if torch.sum(n) > 0 and update_prototype is True:
            f = F.normalize(f, p=2, dim=-1)

            new_value = momentum_update(old_value=protos[k, n != 0, :], new_value=f[n != 0, :],
                                        momentum=coefficient, debug=False)
            protos[k, n != 0, :] = new_value

        proto_target[gt_seg == k] = indexs + (num_prototype * k)

    # prototypes = nn.Parameter(F.normalize(protos, p=2, dim=-1),
    #                                 requires_grad=False)

    # if dist.is_available() and dist.is_initialized():
    #     protos = prototypes.data.clone()
    #     dist.all_reduce(protos.div_(dist.get_world_size()))
    #     prototypes = nn.Parameter(protos, requires_grad=False)

    # rearrange_logit = torch.zeros_like(proto_logits)
    # for i in range(0, num_prototype):
    #     rearrange_logit[:, i::num_prototype] = proto_logits[:, i*num_unify_classes:(i+1)*num_unify_classes]

    return proto_logits, proto_target, protos


def KmeansProtoLearning(configer, memory_bank, memory_bank_ptr, _c, cluster_seg, gt_seg):
    num_unify_classes = configer.get('num_unify_classes')
    num_prototype = configer.get('contrast', 'num_prototype')
    coefficient = configer.get('contrast', 'coefficient')
    network_stride = configer.get('network', 'stride')
                
    cluster_centers = torch.mean(memory_bank, dim=1)
    choice_cluster, initial_state = kmeans(_c[cluster_seg], num_unify_classes, cluster_centers=cluster_centers, distance='cosine', device='cpu', memory_bank=memory_bank, constraint_matrix=gt_seg)
    return choice_cluster, initial_state
    
    
if __name__ == '__main__':
    

    from tools.configer import Configer
    configer = Configer(configs='configs/test_kmeans.json')
    memory_bank = torch.tensor([[[1,1],
                                 [2,2]],
                                [[-1,1],
                                 [-2,1]],
                                [[0,-1],
                                 [1,-2]]], dtype=torch.float)
    memory_bank_ptr = torch.tensor([0,0,0])
    _c = torch.tensor([[0,0],
                       [1,2],
                       [-1,2],
                       [-1,-2],
                       [1,1],
                       [1,1]], dtype=torch.float)
    cluster_seg = torch.ones(6, dtype=torch.bool)
    gt_seg = torch.tensor([[1,1,1],
                            [1,1,1],
                            [1,1,1],
                            [1,1,1],
                            [0,1,1],
                            [1,0,1]], dtype=torch.bool).logical_not()
    choice_cluster, initial_state = KmeansProtoLearning(configer, memory_bank, memory_bank_ptr, _c, cluster_seg, gt_seg)
    print(choice_cluster)
    print(initial_state)
    