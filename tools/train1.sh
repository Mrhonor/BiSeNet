export CUDA_VISIBLE_DEVICES=0,1,2
python -m torch.distributed.launch \
--nproc_per_node=3 --master_port 16852 tools/train_amp_contrast_single.py \
