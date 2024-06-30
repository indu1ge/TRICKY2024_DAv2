export CUDA_VISIBLE_DEVICES=1
python generate_with_maskcolored.py \
    --encoder vitg \
    --img_path data/tricky/test_tricky_nogt \
    --mask_path data/tricky/test_tricky_nogt_mask \
    --outdir data/tricky/test_output/vitg_std_maskcolored \
    --ckpt checkpoints/depth_anything_v2_vitg.pth