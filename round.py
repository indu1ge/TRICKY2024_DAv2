import os
import glob
import logging
from tqdm import tqdm
import numpy as np

EXTENSION_LIST = ['npy']


input_npy_dir = '/zssd/szy/Depth-Anything-V2/data/tricky/test_output/vitg_bs1_742_lr5e-7_mse_valid_mask_epoch100/epoch_99/npy_ensemble_light'
output_npy_dir = '/zssd/szy/Depth-Anything-V2/data/tricky/test_output/vitg_bs1_742_lr5e-7_mse_valid_mask_epoch100/epoch_99/npy_ensemble_light_trunc3'
os.makedirs(output_npy_dir)

for class_folder in os.listdir(input_npy_dir):
    class_folder_path = os.path.join(input_npy_dir, class_folder)
    output_class_path = os.path.join(output_npy_dir, class_folder)
    

    if os.path.isdir(class_folder_path):
        os.makedirs(output_class_path)
        rgb_filename_list = glob.glob(os.path.join(class_folder_path, "*"))
        # rgb_filename_list = [
        #     f for f in rgb_filename_list if os.path.splitext(f)[1].lower() in EXTENSION_LIST
        # ]
        rgb_filename_list = sorted(rgb_filename_list)
        n_images = len(rgb_filename_list)
        if n_images > 0:
            logging.info(f"Found {n_images} images")
        else:
            logging.error(f"No image found in '{class_folder_path}'")
            exit(1)
        
        for input_image_path0 in tqdm(rgb_filename_list, desc=f"Estimating depth", leave=True):
            npy_pred_float32 = np.load(input_image_path0)
            npy_pred_trunc4 = np.round(npy_pred_float32, 3)
            # npy_pred_trunc4 = npy_pred_float32.astype(np.float16)

            rgb_name_base = os.path.splitext(os.path.basename(input_image_path0))[0]
            pred_name_base = rgb_name_base
            
            npy_save_path = os.path.join(output_class_path, f"{pred_name_base}.npy")
            if os.path.exists(npy_save_path):
                logging.warning(f"Existing file: '{npy_save_path}' will be overwritten")
            np.save(npy_save_path, npy_pred_trunc4)
