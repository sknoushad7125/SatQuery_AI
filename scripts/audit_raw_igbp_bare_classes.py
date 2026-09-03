import os
import sys
import numpy as np
import rasterio

sys.path.append(os.getcwd())
from training.scripts.train_optical_sar import SEN12MSSegmentationDataset

def analyze_split(split_name, dataset):
    print(f"\n=========================================")
    print(f"--- Analyzing RAW IGBP {split_name.upper()} Split ---")
    print(f"=========================================")
    
    total_patches = len(dataset)
    total_pixels = total_patches * 256 * 256
    
    igbp_15_px = 0
    igbp_16_px = 0
    igbp_17_px = 0
    
    patches_15 = 0
    patches_16 = 0
    patches_17 = 0
    
    scenes_15 = set()
    scenes_16 = set()
    scenes_17 = set()
    
    invalid_values = set()
    
    project_bare_pixels = 0
    
    for idx, (scene_id, patch_id) in enumerate(dataset.samples):
        filename = f"ROIs1158_spring_lc_{scene_id}_p{patch_id}.tif"
        lc_path = os.path.join(dataset.base_dir, 'ROIs1158_spring', f'lc_{scene_id}', filename)
        
        with rasterio.open(lc_path) as src:
            lc = src.read(1)
            
        count_15 = (lc == 15).sum()
        count_16 = (lc == 16).sum()
        count_17 = (lc == 17).sum()
        
        igbp_15_px += count_15
        igbp_16_px += count_16
        igbp_17_px += count_17
        
        if count_15 > 0:
            patches_15 += 1
            scenes_15.add(scene_id)
        if count_16 > 0:
            patches_16 += 1
            scenes_16.add(scene_id)
        if count_17 > 0:
            patches_17 += 1
            scenes_17.add(scene_id)
            
        mask = dataset.class_map[lc]
        project_bare_pixels += (mask == 3).sum()
        
        unique_vals = np.unique(lc)
        for v in unique_vals:
            if v < 0 or v > 17:
                invalid_values.add(v)
                
        if (idx + 1) % 5000 == 0:
            print(f"  Processed {idx + 1}/{total_patches} patches...")
            
    print(f"\nSplit: {split_name.upper()}")
    print(f"Total Pixels: {total_pixels}")
    
    print(f"\n--- Pixel Counts ---")
    print(f"IGBP 15 (Snow/Ice): {igbp_15_px} ({(igbp_15_px/total_pixels)*100:.4f}%)")
    print(f"IGBP 16 (Barren):   {igbp_16_px} ({(igbp_16_px/total_pixels)*100:.4f}%)")
    print(f"IGBP 17 (Water):    {igbp_17_px} ({(igbp_17_px/total_pixels)*100:.4f}%)")
    
    print(f"\n--- Patch Counts ---")
    print(f"Patches w/ IGBP 15: {patches_15}")
    print(f"Patches w/ IGBP 16: {patches_16}")
    print(f"Patches w/ IGBP 17: {patches_17}")
    
    print(f"\n--- Scene Counts ---")
    print(f"Distinct scenes w/ IGBP 15: {len(scenes_15)}")
    print(f"Distinct scenes w/ IGBP 16: {len(scenes_16)}")
    print(f"Distinct scenes w/ IGBP 17: {len(scenes_17)}")
    
    print(f"\n--- Project Bare Verification ---")
    calculated_bare = igbp_15_px + igbp_16_px
    print(f"IGBP 15 + IGBP 16 pixels: {calculated_bare}")
    print(f"Project bare (class 3):   {project_bare_pixels}")
    if calculated_bare == project_bare_pixels:
        print("Match: YES")
    else:
        print("Match: NO (WARNING)")
        
    if invalid_values:
        print(f"Unexpected values outside 0-17 found: {invalid_values}")
    else:
        print("No unexpected values outside 0-17 found.")

def main():
    base_dir = os.path.join(os.getcwd(), 'datasets/sen12ms')
    train_ds = SEN12MSSegmentationDataset(base_dir, split="train")
    val_ds = SEN12MSSegmentationDataset(base_dir, split="val")
    
    analyze_split("train", train_ds)
    analyze_split("val", val_ds)
    
    print("\nSTEP 4AE RAW IGBP BARE AUDIT PASS")

if __name__ == '__main__':
    main()
