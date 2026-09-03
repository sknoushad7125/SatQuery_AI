import os
import glob
import json
import numpy as np
from PIL import Image
from tqdm import tqdm

def process_levir():
    print("Processing LEVIR-CD...")
    splits = ["train", "val", "test"]
    base_dir = "datasets/levir_cd"
    out_dir = "datasets/processed/manifests"
    os.makedirs(out_dir, exist_ok=True)
    
    manifest = {
        "dataset": "LEVIR-CD",
        "stats": {
            "total_pairs": 0,
            "total_pixels": 0,
            "changed_pixels": 0,
            "empty_masks": 0,
            "split_counts": {"train": 0, "val": 0, "test": 0}
        },
        "records": {}
    }
    
    for split in splits:
        print(f"Validating split: {split}")
        a_dir = os.path.join(base_dir, split, "A")
        b_dir = os.path.join(base_dir, split, "B")
        label_dir = os.path.join(base_dir, split, "label")
        
        if not os.path.exists(a_dir): continue
        
        files = os.listdir(a_dir)
        for f in tqdm(files):
            if not f.endswith(".png"): continue
            
            path_a = os.path.join(a_dir, f)
            path_b = os.path.join(b_dir, f)
            path_lbl = os.path.join(label_dir, f)
            
            if not (os.path.exists(path_b) and os.path.exists(path_lbl)):
                print(f"[ERROR] Missing B or label for {f}")
                continue
                
            with Image.open(path_a) as imga, Image.open(path_b) as imgb, Image.open(path_lbl) as imglbl:
                if imga.size != (1024, 1024) or imgb.size != (1024, 1024) or imglbl.size != (1024, 1024):
                    print(f"[WARNING] Dimension mismatch for {f}")
                    continue
                if imga.mode != "RGB" or imgb.mode != "RGB":
                    print(f"[WARNING] Not RGB for {f}")
                    continue
                
                lbl_arr = np.array(imglbl)
                # Ensure binary (sometimes it's 255)
                lbl_arr = (lbl_arr > 127).astype(np.uint8)
                changed = lbl_arr.sum()
                is_empty = (changed == 0)
                
                manifest["stats"]["total_pairs"] += 1
                manifest["stats"]["split_counts"][split] += 1
                manifest["stats"]["total_pixels"] += (1024 * 1024)
                manifest["stats"]["changed_pixels"] += int(changed)
                if is_empty:
                    manifest["stats"]["empty_masks"] += 1
                    
                manifest["records"][f] = {
                    "split": split,
                    "image_a": path_a,
                    "image_b": path_b,
                    "label": path_lbl,
                    "dimensions": [1024, 1024],
                    "changed_pixels": int(changed),
                    "is_empty": bool(is_empty)
                }
                
    manifest["stats"]["changed_ratio"] = manifest["stats"]["changed_pixels"] / max(1, manifest["stats"]["total_pixels"])
    
    with open(os.path.join(out_dir, "levir_cd_manifest.json"), "w") as f:
        json.dump(manifest, f, indent=2)
    print("LEVIR-CD Manifest generated.")

def process_rsvqa():
    print("Processing RSVQA-LR...")
    # Just a simple check for RSVQA
    out_dir = "datasets/processed/manifests"
    os.makedirs(out_dir, exist_ok=True)
    manifest = {"dataset": "RSVQA-LR", "records": {}}
    # RSVQA processing logic will be light for now as VQA is phase 3
    with open(os.path.join(out_dir, "rsvqa_lr_manifest.json"), "w") as f:
        json.dump(manifest, f)
        
def process_vrsbench():
    print("Processing VRSBench...")
    out_dir = "datasets/processed/manifests"
    os.makedirs(out_dir, exist_ok=True)
    manifest = {"dataset": "VRSBench", "records": {}}
    with open(os.path.join(out_dir, "vrsbench_manifest.json"), "w") as f:
        json.dump(manifest, f)

if __name__ == "__main__":
    process_levir()
    process_rsvqa()
    process_vrsbench()
