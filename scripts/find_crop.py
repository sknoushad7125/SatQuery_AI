import os
import numpy as np
from PIL import Image

def load_image(path):
    img = Image.open(path).convert('RGB')
    return np.array(img)

def find_crop():
    crop_path = "datasets/cdvqa_sample/cdvqa-test-00000000.0.img"
    if not os.path.exists(crop_path):
        print("Crop not found")
        return
        
    crop = load_image(crop_path)
    crop_h, crop_w = crop.shape[:2]
    
    base_dir = "datasets/levir_cd"
    splits = ["train", "val", "test"]
    
    print(f"Searching for crop of size {crop_h}x{crop_w}...")
    
    for split in splits:
        dir_a = os.path.join(base_dir, split, "A")
        if not os.path.exists(dir_a): continue
        
        for fname in os.listdir(dir_a):
            if not fname.endswith(('.png', '.tif', '.jpg')): continue
            
            img_path = os.path.join(dir_a, fname)
            img = load_image(img_path)
            
            # Since we suspect it's a grid crop, check grid positions first
            # e.g., for 512x512 crops in 1024x1024, there are 4 grid positions
            found = False
            for y in range(0, img.shape[0] - crop_h + 1, crop_h):
                for x in range(0, img.shape[1] - crop_w + 1, crop_w):
                    img_crop = img[y:y+crop_h, x:x+crop_w]
                    if np.array_equal(img_crop, crop):
                        print(f"EXACT MATCH FOUND! Scene: {split}/{fname}, x:{x}, y:{y}")
                        found = True
                        break
                if found: break
            if found:
                # Let's also check B image
                # The crop we have is .0.img which is likely T1 (A). .1.img is T2 (B).
                # We can verify .1.img against B.
                crop_b = load_image("datasets/cdvqa_sample/cdvqa-test-00000000.1.img")
                img_b_path = os.path.join(base_dir, split, "B", fname)
                img_b = load_image(img_b_path)
                img_b_crop = img_b[y:y+crop_h, x:x+crop_w]
                if np.array_equal(img_b_crop, crop_b):
                    print("B image also matches exactly!")
                return

    print("No exact grid match found. Performing sliding window search (this will be slow)...")
    # For speed, just check every 16 pixels
    for split in splits:
        dir_a = os.path.join(base_dir, split, "A")
        if not os.path.exists(dir_a): continue
        
        for fname in os.listdir(dir_a):
            if not fname.endswith(('.png', '.tif', '.jpg')): continue
            
            img_path = os.path.join(dir_a, fname)
            img = load_image(img_path)
            
            for y in range(0, img.shape[0] - crop_h + 1, 16):
                for x in range(0, img.shape[1] - crop_w + 1, 16):
                    img_crop = img[y:y+crop_h, x:x+crop_w]
                    # Check first row quickly
                    if np.array_equal(img_crop[0], crop[0]):
                        if np.array_equal(img_crop, crop):
                            print(f"SLIDING MATCH FOUND! Scene: {split}/{fname}, x:{x}, y:{y}")
                            return

if __name__ == "__main__":
    find_crop()
