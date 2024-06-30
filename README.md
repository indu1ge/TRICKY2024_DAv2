Our method is based on Depth Anything V2. By finetuning on the Booster training set, our method achieves 90.61 in delta 1.05 for Transparent/Mirror Surfaces areas.


## Usage

### Data Preparation
```
|-data
  |-tricky
    |-test_tricky_nogt
      |-Appliances
      |-Balance
        ...
|-ckpt
  |-weights
```

### Installation

```bash
conda create -n Marigold python==3.10
conda activate Marigold
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
pip3 install -U xformers --index-url https://download.pytorch.org/whl/cu118
pip install opencv-python transformers matplotlib safetensors accelerate tensorboard datasets scipy einops pytorch_lightning omegaconf diffusers peft
pip3 install h5py scikit-image tqdm bitsandbytes wandb tabulate
```

### Evaluate
First, you need to download the pretrained checkpoint to 'ckpt/'.

```bash
# generate the initial depth predictions
bash generate.sh
# ensemble the depth predictions under different lightning conditions
python ensemble_light.py
# truncate the predictions to three decimal places to meet the maximum submission size on Codalab
python round.py
# apply medium filtering for better results
python filter.py
# zip the .npy files into submission.zip
cd data/tricky/test_output/final/npy_ensemble_light_trunc3_mediumfilter_5
zip -r submission.zip ./*
```


## LICENSE

Depth-Anything-V2-Small model is under the Apache-2.0 license. Depth-Anything-V2-Base/Large/Giant models are under the CC-BY-NC-4.0 license.


## Citation

If you find this project useful, please consider citing:

```bibtex
@article{depth_anything_v2,
  title={Depth Anything V2},
  author={Yang, Lihe and Kang, Bingyi and Huang, Zilong and Zhao, Zhen and Xu, Xiaogang and Feng, Jiashi and Zhao, Hengshuang},
  journal={arXiv:2406.09414},
  year={2024}
}

@inproceedings{depth_anything_v1,
  title={Depth Anything: Unleashing the Power of Large-Scale Unlabeled Data}, 
  author={Yang, Lihe and Kang, Bingyi and Huang, Zilong and Xu, Xiaogang and Feng, Jiashi and Zhao, Hengshuang},
  booktitle={CVPR},
  year={2024}
}
```