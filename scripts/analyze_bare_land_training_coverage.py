import os
import sys
import numpy as np
import rasterio

sys.path.append(os.getcwd())
from training.scripts.train_optical_sar import SEN12MSSegmentationDataset

def main():
    print("Starting Bare Land Coverage Audit...")
    
    base_dir = os.path.join(os.getcwd(), 'datasets/sen12ms')
    train_ds = SEN12MSSegmentationDataset(base_dir, split="train")
    class_map = train_ds.class_map
    
    total_train_patches = len(train_ds)
    patch_size_pixels = 256 * 256
    
    # Stats tracking
    total_bare_pixels = 0
    total_pixels_all = total_train_patches * patch_size_pixels
    
    patches_gt_0 = 0
    patches_gt_10 = 0
    patches_gt_01_pct = 0
    patches_gt_1_pct = 0
    patches_gt_5_pct = 0
    
    bare_rich_patches = []
    all_bare_patches = []
    
    scene_ids_with_bare = set()
    bare_rich_scene_counts = {}
    
    for idx, (scene_id, patch_id) in enumerate(train_ds.samples):
        filename = f"ROIs1158_spring_lc_{scene_id}_p{patch_id}.tif"
        lc_path = os.path.join(train_ds.base_dir, 'ROIs1158_spring', f'lc_{scene_id}', filename)
        
        with rasterio.open(lc_path) as src:
            lc = src.read(1)
            
        mask = class_map[lc]
        bare_px = (mask == 3).sum()
        
        if bare_px > 0:
            total_bare_pixels += bare_px
            pct = bare_px / patch_size_pixels * 100.0
            
            patches_gt_0 += 1
            if bare_px >= 10: patches_gt_10 += 1
            if pct >= 0.1: patches_gt_01_pct += 1
            if pct >= 1.0: 
                patches_gt_1_pct += 1
                bare_rich_patches.append(pct)
                bare_rich_scene_counts[scene_id] = bare_rich_scene_counts.get(scene_id, 0) + 1
            if pct >= 5.0: patches_gt_5_pct += 1
            
            all_bare_patches.append((pct, bare_px, lc_path, scene_id))
            scene_ids_with_bare.add(scene_id)
            
        if (idx + 1) % 5000 == 0:
            print(f"  Processed {idx + 1}/{total_train_patches} patches...")
            
    bare_pct_overall = (total_bare_pixels / total_pixels_all) * 100.0
    
    print("\n--- 1. Patch Counts ---")
    print(f"Total training patches containing bare land (>= 1 pixel): {patches_gt_0}")
    print(f"Total patches with bare >= 10 pixels:  {patches_gt_10}")
    print(f"Total patches with bare >= 0.1%:       {patches_gt_01_pct}")
    print(f"Total patches with bare >= 1.0%:       {patches_gt_1_pct}")
    print(f"Total patches with bare >= 5.0%:       {patches_gt_5_pct}")
    
    print("\n--- 2. Pixel Statistics ---")
    print(f"Total bare-land pixels in training set: {total_bare_pixels}")
    print(f"Bare-land percentage of all pixels:     {bare_pct_overall:.4f}%")
    
    print("\n--- 3. Bare-Rich Stratum (>= 1%) ---")
    print(f"Number of patches: {len(bare_rich_patches)}")
    if bare_rich_patches:
        print(f"Minimum bare percentage: {np.min(bare_rich_patches):.2f}%")
        print(f"Median bare percentage:  {np.median(bare_rich_patches):.2f}%")
        print(f"Maximum bare percentage: {np.max(bare_rich_patches):.2f}%")
    
    print("\n--- 4. Scene Distribution ---")
    print(f"Distinct scene IDs containing bare land: {len(scene_ids_with_bare)}")
    print("Concentration of bare-rich (>=1%) patches across scenes:")
    for sid, count in sorted(bare_rich_scene_counts.items(), key=lambda x: x[1], reverse=True):
        print(f"  Scene {sid}: {count} bare-rich patches")
        
    print("\n--- 5. Top 20 Patches by Bare Percentage ---")
    all_bare_patches.sort(key=lambda x: x[0], reverse=True)
    for i, (pct, px, path, sid) in enumerate(all_bare_patches[:20]):
        print(f"  {i+1}. {pct:.2f}% ({px} px) -> {path}")
        
    print("\nSTEP 4AA BARE LAND COVERAGE AUDIT PASS")

if __name__ == '__main__':
    main()
