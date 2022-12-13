import torch

def memory_bank_push(configer, memory_bank, memory_bank_ptr, _c, gt_seg):
    num_unify_classes = configer.get('num_unify_classes')
    num_prototype = configer.get('contrast', 'num_prototype')
    

    for i in range(0, num_unify_classes):
        if not (gt_seg==int(i)).any():
            continue
        
        else:
            this_feat = _c[gt_seg==int(i)]
            # pixel enqueue and dequeue
            num_pixel = this_feat.shape[0]
            perm = torch.randperm(num_pixel)
            K = min(num_pixel, num_prototype)
            
            feat = this_feat[perm[:K], :]
            
            ptr = int(memory_bank_ptr[i])

            if ptr + K >= num_prototype:
                remain_num = num_prototype - ptr
                new_cir_num = ptr + K - num_prototype
                memory_bank[i, ptr:, :] = feat[:remain_num]
                memory_bank[i, :new_cir_num, :] = feat[remain_num:]
                memory_bank_ptr[i] = new_cir_num
            else:
                memory_bank[i, ptr:ptr + K, :] = feat
                memory_bank_ptr[i] = (memory_bank_ptr[i] + K) % num_prototype