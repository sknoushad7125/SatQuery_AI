import os
import sys
import numpy as np
import pandas as pd
import rasterio

sys.path.append(os.getcwd())
from training.scripts.train_optical_sar import SEN12MSSegmentationDataset

def main():
    print("Starting Bare Land Visual Diversity Audit...")
    
    base_dir = os.path.join(os.getcwd(), 'datasets/sen12ms')
    train_ds = SEN12MSSegmentationDataset(base_dir, split="train")
    class_map = train_ds.class_map
    
    bare_rich_data = []
    scene_counts = {}
    
    print(f"Scanning {len(train_ds)} training patches for bare_rich patches and computing S1/S2 stats...")
    for idx, (scene_id, patch_id) in enumerate(train_ds.samples):
        lc_file = f"ROIs1158_spring_lc_{scene_id}_p{patch_id}.tif"
        lc_path = os.path.join(base_dir, 'ROIs1158_spring', f'lc_{scene_id}', lc_file)
        
        with rasterio.open(lc_path) as src:
            lc = src.read(1)
        
        mask = class_map[lc]
        bare_pct = (mask == 3).sum() / (256 * 256) * 100.0
        
        if bare_pct >= 1.0:
            s2_file = f"ROIs1158_spring_s2_{scene_id}_p{patch_id}.tif"
            s2_path = os.path.join(base_dir, 'ROIs1158_spring', f's2_{scene_id}', s2_file)
            
            s1_file = f"ROIs1158_spring_s1_{scene_id}_p{patch_id}.tif"
            s1_path = os.path.join(base_dir, 'ROIs1158_spring', f's1_{scene_id}', s1_file)
            
            with rasterio.open(s2_path) as src2:
                s2 = src2.read() # [13, 256, 256]
            with rasterio.open(s1_path) as src1:
                s1 = src1.read() # [2, 256, 256]
            
            patch_stats = {
                'scene_id': scene_id,
                'patch_id': patch_id,
                'bare_percentage': bare_pct
            }
            
            for b in range(13):
                patch_stats[f's2_b{b+1}_mean'] = np.mean(s2[b])
                patch_stats[f's2_b{b+1}_std'] = np.std(s2[b])
                
            patch_stats['s1_vv_mean'] = np.mean(s1[0])
            patch_stats['s1_vv_std'] = np.std(s1[0])
            patch_stats['s1_vh_mean'] = np.mean(s1[1])
            patch_stats['s1_vh_std'] = np.std(s1[1])
            
            bare_rich_data.append(patch_stats)
            scene_counts[scene_id] = scene_counts.get(scene_id, 0) + 1
            
        if (idx + 1) % 5000 == 0:
            print(f"  Processed {idx + 1}/{len(train_ds)} patches...")
            
    df = pd.DataFrame(bare_rich_data)
    
    print("\n--- Aggregated Statistics (across 295 bare-rich patches) ---")
    cols_to_agg = [c for c in df.columns if c not in ['scene_id', 'patch_id']]
    agg_df = df[cols_to_agg].agg(['mean', 'median', 'min', 'max', 'std']).T
    # Reorder columns slightly to match standard stats reporting
    agg_df = agg_df[['mean', 'median', 'min', 'max', 'std']]
    print(agg_df.to_string(float_format="%.4f"))
    
    print("\n--- Scene Distribution ---")
    print(f"Total bare-rich patches: {len(df)}")
    print(f"Unique scenes containing bare-rich patches: {len(scene_counts)}")
    print("Patches per scene:")
    for sid, count in sorted(scene_counts.items(), key=lambda x: x[1], reverse=True):
        print(f"  Scene {sid}: {count} patches")
        
    os.makedirs("training/analysis", exist_ok=True)
    csv_path = "training/analysis/bare_land_visual_diversity.csv"
    df.to_csv(csv_path, index=False)
    print(f"\nSaved detailed patch statistics to: {csv_path}")
    
    print("\nSTEP 4AF BARE VISUAL DIVERSITY AUDIT PASS")

if __name__ == '__main__':
    main()
