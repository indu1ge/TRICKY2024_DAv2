export CUDA_VISIBLE_DEVICES=1
python refine.py \
    --encoder vitg \
    --img_path data/tricky/test_tricky_nogt/Window3/camera_00 \
    --mask_path data/tricky/test_tricky_nogt_mask/Window3/image.png \
    --outdir data/tricky/test_output/refine_folder/Window3 \
    --ckpt checkpoints/depth_anything_v2_vitg.pth