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
    
    print("Building LEVIR-CD 256x256 tile hash database...")
    levir_hashes = {} # hash -> (split, scene, A/B, x, y)
    
    splits = ["train", "val", "test"]
    for split in splits:
        a_dir = os.path.join(levir_dir, split, "A")
        if not os.path.exists(a_dir): continue
        scenes = glob.glob(os.path.join(a_dir, "*.png"))
        for scene_path in scenes:
            scene_name = os.path.basename(scene_path)
            
            for folder in ["A", "B"]:
                img_path = os.path.join(levir_dir, split, folder, scene_name)
                if not os.path.exists(img_path): continue
                
                with Image.open(img_path) as img:
                    arr = np.array(img.convert("RGB"))
                    # 1024x1024 -> 4x4 grid of 256x256
                    for r in range(4):
                        for c in range(4):
                            tile = arr[r*256:(r+1)*256, c*256:(c+1)*256]
                            h = get_img_hash(tile)
                            if h not in levir_hashes:
                                levir_hashes[h] = []
                            levir_hashes[h].append({
                                "split": split,
                                "scene": scene_name,
                                "folder": folder,
                                "y": r*256,
                                "x": c*256
                            })
                            
    print(f"Built database with {len(levir_hashes)} unique LEVIR-CD tile hashes.")
    
    print("Mapping CDVQA images...")
    cdvqa_images = glob.glob(os.path.join(cdvqa_dir, "*.png"))
    
    metadata = {
        "total_unique_test_images": len(cdvqa_images) // 2, # each has _1 and _2
        "levir_derived_count": 0,
        "second_derived_count": 0,
        "unresolved_count": 0
    }
    mappings = {}
    
    # Group by base ID
    base_ids = set([os.path.basename(p).split("_")[0] for p in cdvqa_images])
    
    for base_id in sorted(base_ids):
        path_1 = os.path.join(cdvqa_dir, f"{base_id}_1.png")
        path_2 = os.path.join(cdvqa_dir, f"{base_id}_2.png")
        
        if not os.path.exists(path_1) or not os.path.exists(path_2):
            continue
            
        with Image.open(path_1) as img1, Image.open(path_2) as img2:
            arr1 = np.array(img1.convert("RGB"))
            arr2 = np.array(img2.convert("RGB"))
            
            w, h = img1.size
            
            if w == 512 and h == 512:
                # SECOND derived
                metadata["second_derived_count"] += 1
                mappings[base_id] = {
                    "source": "SECOND",
                    "dimensions": [w, h],
                    "confidence": "high (512x512 geometry)"
                }
            elif w == 256 and h == 256:
                h1 = get_img_hash(arr1)
                h2 = get_img_hash(arr2)
                
                match1 = levir_hashes.get(h1, [])
                match2 = levir_hashes.get(h2, [])
                
                # Find intersection of scene and coordinates
                matched_scene = None
                matched_split = None
                matched_coords = None
                
                for m1 in match1:
                    if m1["folder"] == "A":
                        for m2 in match2:
                            if m2["folder"] == "B" and m1["scene"] == m2["scene"] and m1["x"] == m2["x"] and m1["y"] == m2["y"]:
                                matched_scene = m1["scene"]
                                matched_split = m1["split"]
                                matched_coords = [m1["x"], m1["y"]]
                                break
                    if matched_scene: break
                    
                if matched_scene:
                    metadata["levir_derived_count"] += 1
                    mappings[base_id] = {
                        "source": "LEVIR-CD",
                        "dimensions": [w, h],
                        "source_split": matched_split,
                        "source_scene": matched_scene,
                        "crop_coordinates": matched_coords,
                        "image_difference_metric": "Exact Pixel Hash Match (MD5)",
                        "confidence": "verified"
                    }
                else:
                    # Could be overlapping crops or augmented
                    metadata["unresolved_count"] += 1
                    mappings[base_id] = {
                        "source": "UNKNOWN",
                        "dimensions": [w, h],
                        "confidence": "unverified (no pixel match)"
                    }
            else:
                metadata["unresolved_count"] += 1
                
    with open("datasets/cdvqa/cdvqa_mapping.json", "w") as f:
        json.dump({"metadata": metadata, "mappings": mappings}, f, indent=2)
        
    print(json.dumps(metadata, indent=2))
    
if __name__ == "__main__":
    main()
