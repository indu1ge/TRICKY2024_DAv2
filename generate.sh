export CUDA_VISIBLE_DEVICES=0
python generate.py \
    --encoder vitg \
    --img_path data/tricky/test_tricky_nogt \
    --outdir data/tricky/test_output/vitg_bs1_742_lr5e-7_mse_valid_mask_epoch100/epoch_99 \
    --ckpt metric_depth/log/vitg_bs1_742_lr5e-7_mse_valid_mask_epoch100/epoch_99