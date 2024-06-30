import os
import glob
import logging
from tqdm import tqdm
import numpy as np

img_dir = 'data/tricky/test_output/vitg_std/ensemble_light_npy'
output_depth_dir = 'data/tricky/test_output/vitg_std/ensemble_light_npy_depth'
EXTENSION_LIST = ['npy']

for class_folder in os.listdir(img_dir):
    output_class_folder_path = os.path.join(output_depth_dir, class_folder)
    os.makedirs(output_class_folder_path)
    rgb_filename_list = glob.glob(os.path.join(img_dir, class_folder, "*"))
    # rgb_filename_list = [
    #     f for f in rgb_filename_list if os.path.splitext(f)[1].lower() in EXTENSION_LIST
    # ]
    rgb_filename_list = sorted(rgb_filename_list)
    n_images = len(rgb_filename_list)
    if n_images > 0:
        logging.info(f"Found {n_images} images")
    else:
        logging.error(f"No image found in '{class_folder}'")
        exit(1)

    for input_image_path0 in tqdm(rgb_filename_list, desc=f"Estimating depth", leave=True):
        disp = np.load(input_image_path0)
        # disp = np.clip(disp, 1e-1, 1)
        # # disp.clip(1e-3, 1)
        # depth = 1/disp
        # depth = (depth - depth.min())/(depth.max()-depth.min())
        depth = 1-disp
        rgb_name_base = os.path.splitext(os.path.basename(input_image_path0))[0]
        pred_name_base = rgb_name_base
        
        npy_save_path = os.path.join(output_class_folder_path, f"{pred_name_base}.npy")
        if os.path.exists(npy_save_path):
            logging.warning(f"Existing file: '{npy_save_path}' will be overwritten")
        np.save(npy_save_path, depth)
