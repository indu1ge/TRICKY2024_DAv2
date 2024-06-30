import cv2
import torch
from torch.utils.data import Dataset
from torchvision.transforms import Compose
import numpy as np
import random

from dataset.transform import Resize, NormalizeImage, PrepareForNet, Crop
from torchvision import transforms


class Booster(Dataset):
    def __init__(self, filelist_path, mode, size=(518, 518)):
        
        self.mode = mode
        self.size = size
        try:
            self.brightness = (0.8, 1.2)
            self.contrast = (0.8, 1.2)
            self.saturation = (0.8, 1.2)
            self.hue = (-0.1, 0.1)
            transforms.ColorJitter.get_params(
                self.brightness, self.contrast, self.saturation, self.hue)
        except TypeError:
            self.brightness = 0.2
            self.contrast = 0.2
            self.saturation = 0.2
            self.hue = 0.1
        
        with open(filelist_path, 'r') as f:
            self.filelist = f.read().splitlines()
        
        net_w, net_h = size
        self.transform = Compose([
            Resize(
                width=net_w,
                height=net_h,
                resize_target=True if mode == 'train' else False,
                keep_aspect_ratio=True,
                ensure_multiple_of=14,
                resize_method='lower_bound',
                image_interpolation_method=cv2.INTER_CUBIC,
            ),
            NormalizeImage(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            PrepareForNet(),
        ])
    
    def __getitem__(self, item):
        img_path = self.filelist[item].split(' ')[0]
        depth_path = self.filelist[item].split(' ')[1]
        valid_mask_path = self.filelist[item].split(' ')[2]
        tom_mask_path = self.filelist[item].split(' ')[3]
        
        image = cv2.imread(img_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB) / 255.0  # H x W x 3
        
        # depth = cv2.imread(depth_path, cv2.IMREAD_ANYCOLOR | cv2.IMREAD_ANYDEPTH) / 100.0  # cm to m
        disp = np.load(depth_path)
        depth = self.invert_d(disp, 1, 1)
        depth = depth * 1000.0  # H x W

        mask = cv2.imread(valid_mask_path, -1)

        tom_mask = cv2.imread(tom_mask_path, -1)

        
        sample = self.transform({'image': image, 'depth': depth, 'mask': mask, 'semseg_mask': tom_mask})

        sample['image'] = torch.from_numpy(sample['image'])
        color_aug = transforms.ColorJitter(self.brightness, self.contrast, self.saturation, self.hue)
        if random.random() > 0.5:
            sample['image'] = color_aug(sample['image'])
        sample['depth'] = torch.from_numpy(sample['depth'])
        sample['mask'] = torch.from_numpy(sample['mask']).bool()
        sample['semseg_mask'] = torch.from_numpy(sample['semseg_mask']).bool()
        
        # sample['valid_mask'] = (sample['depth'] <= 80)
        
        sample['image_path'] = img_path
        
        return sample

    def __len__(self):
        return len(self.filelist)
    
    def invert_d(self, d, f, b):
        """Convert disparity to depth and viceversa
        Args:
            d: disparity or depth
            f: focal lenght in pixel
            b: baseline
        """
        d = d.astype(np.float32)
        NaN = d <= 0
        d[NaN] = 1
        d = f * b / d
        d[NaN] = 0
        return d