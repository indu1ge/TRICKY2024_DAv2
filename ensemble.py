import numpy as np
import os
import glob
from tqdm import tqdm
import logging

npy_path1 = '/zssd/szy/Depth-Anything-V2/data/tricky/output/std_g'
npy_path2 = '/zssd/szy/Marigold/tricky/output/std'
outdir = '/zssd/szy/Depth-Anything-V2/data/tricky/output/mix_mari_dag_55'

os.makedirs(outdir, exist_ok=True)

for class_folder in os.listdir(npy_path1):
    class_folder_path = os.path.join(npy_path1, class_folder)

    if os.path.isdir(class_folder_path):
        output_class_folder = os.path.join(outdir, class_folder)
        os.makedirs(output_class_folder, exist_ok=True)

    rgb_filename_list = glob.glob(os.path.join(class_folder_path, "*"))
    rgb_filename_list = sorted(rgb_filename_list)

    for input_image_path0 in tqdm(rgb_filename_list, desc=f"Estimating depth", leave=True):
        rgb_name_base = os.path.splitext(os.path.basename(input_image_path0))[0]
        input_image_path1 = os.path.join(npy_path2, class_folder, f"{rgb_name_base}.npy")

        npy1 = np.load(input_image_path0)
        npy2 = np.load(input_image_path1)

        npy_mix = (npy1 + npy2) / 2.
        npy_save_path = os.path.join(output_class_folder, f"{rgb_name_base}.npy")
        if os.path.exists(npy_save_path):
            logging.warning(f"Existing file: '{npy_save_path}' will be overwritten")
        np.save(npy_save_path, npy_mix)
