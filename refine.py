import argparse
import cv2
import glob
import matplotlib
import numpy as np
import os
import torch
import logging
from tqdm import tqdm
from PIL import Image

from depth_anything_v2.dpt import DepthAnythingV2

EXTENSION_LIST = [".jpg", ".jpeg", ".png"]


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Depth Anything V2')
    
    parser.add_argument('--img_path', type=str)
    parser.add_argument('--mask_path', type=str)
    parser.add_argument('--ckpt', type=str)
    parser.add_argument('--input_size', type=int, default=518)
    parser.add_argument('--outdir', type=str, default='./vis_depth')
    
    parser.add_argument('--encoder', type=str, default='vitl', choices=['vits', 'vitb', 'vitl', 'vitg'])
    
    args = parser.parse_args()
    
    DEVICE = 'cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu'
    
    model_configs = {
        'vits': {'encoder': 'vits', 'features': 64, 'out_channels': [48, 96, 192, 384]},
        'vitb': {'encoder': 'vitb', 'features': 128, 'out_channels': [96, 192, 384, 768]},
        'vitl': {'encoder': 'vitl', 'features': 256, 'out_channels': [256, 512, 1024, 1024]},
        'vitg': {'encoder': 'vitg', 'features': 384, 'out_channels': [1536, 1536, 1536, 1536]}
    }
    
    depth_anything = DepthAnythingV2(**model_configs[args.encoder])
    print('loading model')
    pretrained_dict = torch.load(args.ckpt, map_location='cpu')
    # pretrained_dict1 = {k.replace('module.', ''): v for k, v in pretrained_dict['model'].items()}
    depth_anything.load_state_dict(pretrained_dict)
    print('successfully load model')
    depth_anything = depth_anything.to(DEVICE).eval()
    
    os.makedirs(args.outdir, exist_ok=True)
    
    cmap = matplotlib.colormaps.get_cmap('Spectral_r')
    # mask0 = np.zeros((752, 1028)).astype(bool)

    # for class_folder in os.listdir(args.img_path):
    mask = cv2.imread(args.mask_path, -1).astype(bool)

    output_class_folder = os.path.join(args.outdir, 'npy')
    os.makedirs(output_class_folder, exist_ok=True)
    output_class_folder_colored = os.path.join(args.outdir, 'colored')
    os.makedirs(output_class_folder_colored, exist_ok=True)
    output_class_folder_grey = os.path.join(args.outdir, 'grey')
    os.makedirs(output_class_folder_grey, exist_ok=True)

    rgb_filename_list = glob.glob(os.path.join(args.img_path, "*"))
    rgb_filename_list = [
        f for f in rgb_filename_list if os.path.splitext(f)[1].lower() in EXTENSION_LIST
    ]
    rgb_filename_list = sorted(rgb_filename_list)
    n_images = len(rgb_filename_list)
    if n_images > 0:
        logging.info(f"Found {n_images} images")
    else:
        logging.error(f"No image found in '{args.img_path}'")
        exit(1)
    with torch.no_grad():
        
        for input_image_path0 in tqdm(rgb_filename_list, desc=f"Estimating depth", leave=True):
            raw_image = cv2.imread(input_image_path0)
            preds = []
            for _ in range(5):
                color = np.random.random([3])
                # color = np.array([1,0,0])
                raw_image[mask] = color
                depth = depth_anything.infer_image(raw_image, args.input_size)
                # print(depth.max())
                # print(depth.min())
                # break
                
                depth = (depth - depth.min()) / (depth.max() - depth.min())
                # print(depth.dtype)
                preds.append(depth)
            depth = np.median(np.stack(preds,axis=0), axis=0)

            # savd as npy
            rgb_name_base = os.path.splitext(os.path.basename(input_image_path0))[0]
            pred_name_base = rgb_name_base
            
            npy_save_path = os.path.join(output_class_folder, f"{pred_name_base}.npy")
            if os.path.exists(npy_save_path):
                logging.warning(f"Existing file: '{npy_save_path}' will be overwritten")
            np.save(npy_save_path, depth)

            # Save as 16-bit uint png
            
            depth = (depth - depth.min()) / (depth.max() - depth.min())
            depth_to_save = (depth * 65535.0).astype(np.uint16)
            png_save_path = os.path.join(output_class_folder_grey, f"{pred_name_base}.png")
            if os.path.exists(png_save_path):
                logging.warning(f"Existing file: '{png_save_path}' will be overwritten")
            Image.fromarray(depth_to_save).save(png_save_path, mode="I;16")

            # Colorize
            colored_save_path = os.path.join(
                output_class_folder_colored, f"{pred_name_base}_colored.png"
            )
            depth_colored = (cmap(depth)[:, :, :3] * 255)[:, :, ::-1].astype(np.uint8)
            # print(depth_colored.shape)
            # print(depth_colored.dtype)
            if os.path.exists(colored_save_path):
                logging.warning(
                    f"Existing file: '{colored_save_path}' will be overwritten"
                )
            Image.fromarray(depth_colored).save(colored_save_path)