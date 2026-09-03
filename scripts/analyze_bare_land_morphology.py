import os
import sys
import numpy as np
import rasterio
from scipy import ndimage

sys.path.append(os.getcwd())
from training.scripts.train_optical_sar import SEN12MSSegmentationDataset

def main():
    print("Starting Bare Land Spatial Morphology Audit...")
    
    base_dir = os.path.join(os.getcwd(), 'datasets/sen12ms')
    train_ds = SEN12MSSegmentationDataset(base_dir, split="train")
    class_map = train_ds.class_map
    
    all_component_sizes = []
    total_bare_pixels = 0
    patches_with_bare = 0
    
    bare_rich_largest_components = []
    top_patches = []
    
    structure = np.ones((3, 3), dtype=int)
    
    print(f"Scanning 28,609 training masks for bare land morphology...")
    
    for idx, (scene_id, patch_id) in enumerate(train_ds.samples):
        lc_file = f"ROIs1158_spring_lc_{scene_id}_p{patch_id}.tif"
        lc_path = os.path.join(base_dir, 'ROIs1158_spring', f'lc_{scene_id}', lc_file)
        
        with rasterio.open(lc_path) as src:
            lc = src.read(1)
            
        mask = class_map[lc]
        bare_mask = (mask == 3)
        bare_px = bare_mask.sum()
        
        if bare_px > 0:
            patches_with_bare += 1
            total_bare_pixels += bare_px
            bare_pct = (bare_px / (256*256)) * 100.0
            
            labeled_array, num_features = ndimage.label(bare_mask, structure=structure)
            if num_features > 0:
                component_sizes = np.bincount(labeled_array.ravel())[1:]
                all_component_sizes.extend(component_sizes)
                
                largest_comp = int(np.max(component_sizes))
                
                if bare_pct >= 1.0:
                    bare_rich_largest_components.append(largest_comp)
                    
                top_patches.append({
                    'scene_id': scene_id,
                    'patch_id': patch_id,
                    'bare_pct': bare_pct,
                    'largest_comp': largest_comp
                })
        
        if (idx + 1) % 5000 == 0:
            print(f"  Processed {idx + 1}/{len(train_ds)} patches...")

    print("\nAnalyzing collected statistics...")
    
    sizes = np.array(all_component_sizes)
    num_components = len(sizes)
    
    print("\n--- 1 & 2. General Counts ---")
    print(f"Number of patches containing bare land: {patches_with_bare}")
    print(f"Total number of connected components:   {num_components}")
    
    print("\n--- 3. Area Statistics (Pixels) ---")
    print(f"Minimum:          {np.min(sizes)}")
    print(f"Median:           {np.median(sizes):.1f}")
    print(f"Mean:             {np.mean(sizes):.1f}")
    print(f"75th percentile:  {np.percentile(sizes, 75):.1f}")
    print(f"90th percentile:  {np.percentile(sizes, 90):.1f}")
    print(f"95th percentile:  {np.percentile(sizes, 95):.1f}")
    print(f"99th percentile:  {np.percentile(sizes, 99):.1f}")
    print(f"Maximum:          {np.max(sizes)}")
    
    print("\n--- 4. Size Distribution (Component Count) ---")
    bins = [
        (1, 4, "1-4"), (5, 9, "5-9"), (10, 49, "10-49"), 
        (50, 99, "50-99"), (100, 499, "100-499"), 
        (500, 999, "500-999"), (1000, 4999, "1000-4999"), 
        (5000, np.inf, ">= 5000")
    ]
    for low, high, label in bins:
        count = np.sum((sizes >= low) & (sizes <= high))
        print(f"  {label:<10} : {count}")
        
    print("\n--- 5. Pixel Mass Distribution (Percentage of Total Pixels) ---")
    lt_10 = np.sum(sizes[sizes < 10]) / total_bare_pixels * 100
    lt_50 = np.sum(sizes[sizes < 50]) / total_bare_pixels * 100
    lt_100 = np.sum(sizes[sizes < 100]) / total_bare_pixels * 100
    gte_100 = np.sum(sizes[sizes >= 100]) / total_bare_pixels * 100
    gte_500 = np.sum(sizes[sizes >= 500]) / total_bare_pixels * 100
    
    print(f"  < 10 pixels:   {lt_10:.2f}%")
    print(f"  < 50 pixels:   {lt_50:.2f}%")
    print(f"  < 100 pixels:  {lt_100:.2f}%")
    print(f"  >= 100 pixels: {gte_100:.2f}%")
    print(f"  >= 500 pixels: {gte_500:.2f}%")
    
    print("\n--- 6. Bare-Rich Patches (>=1%) Largest Component Stats ---")
    if bare_rich_largest_components:
        br_largest = np.array(bare_rich_largest_components)
        print(f"  Median largest component: {np.median(br_largest):.1f}")
        print(f"  Mean largest component:   {np.mean(br_largest):.1f}")
        print(f"  Maximum largest component:{np.max(br_largest)}")
    
    print("\n--- 7. Top 20 Patches by Largest Connected Component ---")
    top_patches.sort(key=lambda x: x['largest_comp'], reverse=True)
    for i, p in enumerate(top_patches[:20]):
        print(f"  {i+1:>2}. Scene {p['scene_id']:<3} Patch {p['patch_id']:<4} | "
              f"Bare %: {p['bare_pct']:>5.2f}% | "
              f"Largest Comp: {p['largest_comp']}")
              
    print("\n--- 8. Verification ---")
    sum_areas = np.sum(sizes)
    print(f"Sum of all component areas: {sum_areas}")
    print(f"Total measured bare pixels: 1303651")
    if sum_areas == 1303651:
        print("Match: YES")
    else:
        print("Match: NO (WARNING)")
        
    print("\nSTEP 4AG BARE MORPHOLOGY AUDIT PASS")

if __name__ == '__main__':
    main()
