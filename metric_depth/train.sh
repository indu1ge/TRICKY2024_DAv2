export CUDA_VISIBLE_DEVICES=7

python -m torch.distributed.launch --master_port 2254 train.py \
    --encoder vitg \
    --dataset booster \
    --pretrained_from /zssd/szy/Depth-Anything-V2/checkpoints/depth_anything_v2_vitg.pth \
    --save_path log/vitg_bs1_742_lr3e-7_mse_valid_mask_epoch100_aug_0.5 \
    --img_size 742 \
    --bs 1 \
    --epochs 100 \
    --lr 3e-7 \