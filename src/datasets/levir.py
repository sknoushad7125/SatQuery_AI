import os
import json
import torch
import numpy as np
from PIL import Image
from torch.utils.data import Dataset
import albumentations as A
from albumentations.pytorch import ToTensorV2

class LEVIRCDDataset(Dataset):
    def __init__(self, manifest_path, split="train", crop_size=256, is_train=True):
        super().__init__()
        with open(manifest_path, "r") as f:
            manifest = json.load(f)
            
        self.records = [rec for rec in manifest["records"].values() if rec["split"] == split]
        self.is_train = is_train
        self.crop_size = crop_size
        
        # Albumentations for matched transforms
        if self.is_train:
            self.transform = A.Compose([
                A.RandomCrop(width=crop_size, height=crop_size),
                A.HorizontalFlip(p=0.5),
                A.VerticalFlip(p=0.5),
                A.RandomRotate90(p=0.5),
                A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
                ToTensorV2(),
            ], additional_targets={'image0': 'image'})
        else:
            self.transform = A.Compose([
                A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
                ToTensorV2(),
            ], additional_targets={'image0': 'image'})
            
    def __len__(self):
        return len(self.records)
        
    def __getitem__(self, idx):
        rec = self.records[idx]
        
        img_a = np.array(Image.open(rec["image_a"]).convert("RGB"))
        img_b = np.array(Image.open(rec["image_b"]).convert("RGB"))
        lbl = np.array(Image.open(rec["label"]))
        lbl = (lbl > 127).astype(np.float32) # Binary mask
        
        transformed = self.transform(image=img_a, image0=img_b, mask=lbl)
        
        # Output: [C, H, W]
        t_img_a = transformed["image"]
        t_img_b = transformed["image0"]
        t_lbl = transformed["mask"].unsqueeze(0) # [1, H, W]
        
        return {
            "image_a": t_img_a,
            "image_b": t_img_b,
            "mask": t_lbl,
            "filename": os.path.basename(rec["image_a"])
        }
