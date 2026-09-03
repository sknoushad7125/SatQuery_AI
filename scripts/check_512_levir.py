import os
import json
import glob
import numpy as np
from PIL import Image
import hashlib

def get_img_hash(img_array):
    return hashlib.md5(img_array.tobytes()).hexdigest()

def main():
    levir_dir = "datasets/levir_cd"
    cdvqa_dir = "datasets/cdvqa/images"
    
    print("Building LEVIR-CD 512x512 tile hash database...")
    levir_hashes_512 = {}
    levir_hashes_256 = {}
    
    splits = ["train", "val", "test"]
    for split in splits:
        a_dir = os.path.join(levir_dir, split, "A")
        if not os.path.exists(a_dir): continue
        scenes = glob.glob(os.path.join(a_dir, "*.png"))
        for scene_path in scenes:
            scene_name = os.path.basename(scene_path)
            
            img_path_a = os.path.join(levir_dir, split, "A", scene_name)
            img_path_b = os.path.join(levir_dir, split, "B", scene_name)
            if not os.path.exists(img_path_a) or not os.path.exists(img_path_b): continue
            
            with Image.open(img_path_a) as imga, Image.open(img_path_b) as imgb:
                arra = np.array(imga.convert("RGB"))
                arrb = np.array(imgb.convert("RGB"))
                
                # 512x512 crops (4 per image)
                for r in range(2):
                    for c in range(2):
                        ta = arra[r*512:(r+1)*512, c*512:(c+1)*512]
                        ha = get_img_hash(ta)
                        levir_hashes_512[ha] = {"split": split, "scene": scene_name, "folder": "A", "y": r*512, "x": c*512}
                        
                        tb = arrb[r*512:(r+1)*512, c*512:(c+1)*512]
                        hb = get_img_hash(tb)
                        levir_hashes_512[hb] = {"split": split, "scene": scene_name, "folder": "B", "y": r*512, "x": c*512}
                        
    print(f"Built database with {len(levir_hashes_512)} 512x512 hashes.")
    
    cdvqa_images = glob.glob(os.path.join(cdvqa_dir, "*.png"))
    base_ids = set([os.path.basename(p).split("_")[0] for p in cdvqa_images])
    
    levir_matches = 0
    second_inferred = 0
    
    for base_id in base_ids:
        path_1 = os.path.join(cdvqa_dir, f"{base_id}_1.png")
        if not os.path.exists(path_1): continue
        
        with Image.open(path_1) as img1:
            arr1 = np.array(img1.convert("RGB"))
            if img1.size == (512, 512):
                h1 = get_img_hash(arr1)
                if h1 in levir_hashes_512:
                    levir_matches += 1
                else:
                    # Is it resized? Let's try resizing to 256
                    arr_rs = np.array(img1.resize((256,256), Image.BILINEAR).convert("RGB"))
                    h_rs = get_img_hash(arr_rs)
                    if h_rs in levir_hashes_256:
                        levir_matches += 1
                    else:
                        second_inferred += 1
                        
    print(f"LEVIR Matches (512x512): {levir_matches}")
    print(f"SECOND Inferred: {second_inferred}")

if __name__ == "__main__":
    main()
