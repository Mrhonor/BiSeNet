export CUDA_VISIBLE_DEVICES=1,2,3
python -m torch.distributed.run \
--nproc_per_node=3 --master_port 16854 tools/train_amp_contrast_single.py \
