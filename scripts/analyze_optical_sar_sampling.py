import os
import sys
import numpy as np
import rasterio

sys.path.append(os.getcwd())
from training.scripts.train_optical_sar import SEN12MSSegmentationDataset

def analyze_split(split_name, dataset):
    print(f"\n=========================================")
    print(f"--- Analyzing {split_name.upper()} Split ---")
    print(f"=========================================")
    
    strata_stats = {
        'bare_rich': {'veg': [], 'urb': [], 'wat': [], 'bare': [], 'count': 0, 'has_urb': 0, 'has_wat': 0, 'has_bare': 0},
        'water_rich': {'veg': [], 'urb': [], 'wat': [], 'bare': [], 'count': 0, 'has_urb': 0, 'has_wat': 0, 'has_bare': 0},
        'urban_rich': {'veg': [], 'urb': [], 'wat': [], 'bare': [], 'count': 0, 'has_urb': 0, 'has_wat': 0, 'has_bare': 0},
        'mixed_minority': {'veg': [], 'urb': [], 'wat': [], 'bare': [], 'count': 0, 'has_urb': 0, 'has_wat': 0, 'has_bare': 0},
        'vegetation_dominant': {'veg': [], 'urb': [], 'wat': [], 'bare': [], 'count': 0, 'has_urb': 0, 'has_wat': 0, 'has_bare': 0}
    }
    
    class_map = dataset.class_map
    total_patches = len(dataset)
    
    for idx, (scene_id, patch_id) in enumerate(dataset.samples):
        # Correctly formatted string
        filename = f"ROIs1158_spring_lc_{scene_id}_p{patch_id}.tif"
        lc_path = os.path.join(dataset.base_dir, 'ROIs1158_spring', f'lc_{scene_id}', filename)
        
        with rasterio.open(lc_path) as src:
            lc = src.read(1)
            
        mask = class_map[lc]
        total_pixels = float(mask.size)
        
        # Calculate percentages
        p_veg = (mask == 0).sum() / total_pixels * 100.0
        p_urb = (mask == 1).sum() / total_pixels * 100.0
        p_wat = (mask == 2).sum() / total_pixels * 100.0
        p_bare = (mask == 3).sum() / total_pixels * 100.0
        
        # Assign Mutually Exclusive Stratum
        if p_bare >= 1.0:
            stratum = 'bare_rich'
        elif p_wat >= 5.0 and p_bare < 1.0:
            stratum = 'water_rich'
        elif p_urb >= 5.0 and p_wat < 5.0 and p_bare < 1.0:
            stratum = 'urban_rich'
        elif p_urb >= 1.0 or p_wat >= 1.0:
            stratum = 'mixed_minority'
        else:
            stratum = 'vegetation_dominant'
            
        st = strata_stats[stratum]
        st['count'] += 1
        st['veg'].append(p_veg)
        st['urb'].append(p_urb)
        st['wat'].append(p_wat)
        st['bare'].append(p_bare)
        
        if p_urb > 0: st['has_urb'] += 1
        if p_wat > 0: st['has_wat'] += 1
        if p_bare > 0: st['has_bare'] += 1
        
        if (idx + 1) % 5000 == 0 or (idx + 1) == total_patches:
            print(f"  Processed {idx + 1}/{total_patches} patches...")
            
    # Verification
    sum_counts = sum(st['count'] for st in strata_stats.values())
    print(f"\nTotal Patches: {total_patches} | Sum of Strata Counts: {sum_counts}")
    assert total_patches == sum_counts, "ERROR: Strata sum does not match total patches!"
    
    # Reporting
    for name, st in strata_stats.items():
        cnt = st['count']
        pct = (cnt / total_patches) * 100.0
        print(f"\n=> Stratum: {name} | Patches: {cnt} ({pct:.2f}%)")
        
        if cnt > 0:
            print("  Min / Median / Max (%)")
            print(f"    Vegetation: {np.min(st['veg']):.2f} / {np.median(st['veg']):.2f} / {np.max(st['veg']):.2f}")
            print(f"    Built-up:   {np.min(st['urb']):.2f} / {np.median(st['urb']):.2f} / {np.max(st['urb']):.2f}")
            print(f"    Water:      {np.min(st['wat']):.2f} / {np.median(st['wat']):.2f} / {np.max(st['wat']):.2f}")
            print(f"    Bare:       {np.min(st['bare']):.2f} / {np.median(st['bare']):.2f} / {np.max(st['bare']):.2f}")
            print("  Number of patches containing class (> 0%):")
            print(f"    Built-up:   {st['has_urb']}")
            print(f"    Water:      {st['has_wat']}")
            print(f"    Bare:       {st['has_bare']}")
        else:
            print("  Empty Stratum.")

def main():
    base_dir = os.path.join(os.getcwd(), 'datasets/sen12ms')
    print("Loading datasets...")
    train_ds = SEN12MSSegmentationDataset(base_dir, split="train")
    val_ds = SEN12MSSegmentationDataset(base_dir, split="val")
    
    analyze_split("train", train_ds)
    analyze_split("val", val_ds)
    
    print("\nSTEP 4V SAMPLING ANALYSIS PASS")

if __name__ == '__main__':
    main()
