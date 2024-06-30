export CUDA_VISIBLE_DEVICES=0
python generate.py \
    --encoder vitg \
    --img_path data/tricky/test_tricky_nogt \
    --outdir data/tricky/test_output/final \
    --ckpt ckpt/weights