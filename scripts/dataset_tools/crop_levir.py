#!/usr/bin/env python3
import os
from PIL import Image
from pathlib import Path

def crop_split(base_dir, split_name, crop_size=256):
    """
    Takes 1024x1024 original image pairs in a specific split (e.g. train/)
    and crops them into non-overlapping 256x256 tiles to create training patches.
    This guarantees crops from the same original image do not leak across splits.
    """
    split_dir = Path(base_dir) / split_name
    cropped_dir = Path(base_dir) / f"{split_name}_cropped"
    
    if not split_dir.exists():
        return

    for sub in ["A", "B", "label"]:
        (cropped_dir / sub).mkdir(parents=True, exist_ok=True)
        
    a_files = sorted(list((split_dir / "A").glob("*.png")))
    
    total_crops = 0
    for a_path in a_files:
        filename = a_path.name
        b_path = split_dir / "B" / filename
        label_path = split_dir / "label" / filename
        
        if not b_path.exists() or not label_path.exists():
            continue
            
        img_a = Image.open(a_path)
        img_b = Image.open(b_path)
        img_label = Image.open(label_path)
        
        width, height = img_a.size
        
        crop_idx = 0
        for i in range(0, width, crop_size):
            for j in range(0, height, crop_size):
                box = (i, j, i + crop_size, j + crop_size)
                
                crop_a = img_a.crop(box)
                crop_b = img_b.crop(box)
                crop_label = img_label.crop(box)
                
                new_filename = f"{filename.replace('.png', '')}_crop_{crop_idx}.png"
                crop_a.save(cropped_dir / "A" / new_filename)
                crop_b.save(cropped_dir / "B" / new_filename)
                crop_label.save(cropped_dir / "label" / new_filename)
                
                crop_idx += 1
                total_crops += 1
                
    print(f"[{split_name.upper()}] Generated {total_crops} legitimate {crop_size}x{crop_size} crops from {len(a_files)} original scenes.")

if __name__ == "__main__":
    for split in ["train", "val", "test"]:
        crop_split("datasets/levir_cd", split)
