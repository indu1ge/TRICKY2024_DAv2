from metric_depth.util.ensemble import ensemble_depth
import os
import glob
from tqdm import tqdm
import numpy as np
import torch

pred_dir = 'data/tricky/test_output/final/npy'
output_dir = 'data/tricky/test_output/final/npy_ensemble_light'
os.makedirs(output_dir)
for class_folder in os.listdir(pred_dir):
    
    class_path = os.path.join(pred_dir, class_folder)
    if os.path.isdir(class_path):
        rgb_filename_list = glob.glob(os.path.join(class_path, "*"))
        depth_pred_ls = []
        for npy_path in tqdm(rgb_filename_list, desc=f"Ensemble", leave=True):
            npy_pred = np.load(npy_path)
            depth_pred_ls.append(torch.from_numpy(npy_pred).unsqueeze(0).unsqueeze(0))
        depth_pred_ls_cat = torch.concat(depth_pred_ls, dim=0)

        depth_pred_ensembled, _ = ensemble_depth(
            depth_pred_ls_cat,
            scale_invariant=True,
            shift_invariant=True,
        )

        depth_pred_ensembled = depth_pred_ensembled.squeeze()
        output_class_path = os.path.join(output_dir, class_folder)
        os.makedirs(output_class_path)

        for npy_path in tqdm(rgb_filename_list, desc=f"Ensemble", leave=True):
            npy_path_base = os.path.splitext(os.path.basename(npy_path))[0]
            npy_save_path = os.path.join(output_class_path, f"{npy_path_base}.npy")
            np.save(npy_save_path, depth_pred_ensembled)
