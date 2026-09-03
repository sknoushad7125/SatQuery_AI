import sys
import os
import rasterio
import numpy as np
from glob import glob

sys.path.append(os.getcwd())
from sen12ms_dataLoader import SEN12MSDataset, Seasons, S1Bands, S2Bands, LCBands

base_dir = os.path.join(os.getcwd(), 'datasets/sen12ms')
dataset = SEN12MSDataset(base_dir)

s1_files = glob(os.path.join(base_dir, 'ROIs1158_spring', 's1_*', '*.tif'))
s2_files = glob(os.path.join(base_dir, 'ROIs1158_spring', 's2_*', '*.tif'))
lc_files = glob(os.path.join(base_dir, 'ROIs1158_spring', 'lc_*', '*.tif'))

print(f"Total S1 patches: {len(s1_files)}")
print(f"Total S2 patches: {len(s2_files)}")
print(f"Total LC patches: {len(lc_files)}")

# Parse keys (scene_id, patch_id)
def get_key(filepath, prefix):
    # e.g., ROIs1158_spring_s1_1_p100.tif -> (1, 100)
    filename = os.path.basename(filepath)
    parts = filename.replace(".tif", "").split(f"{prefix}_")[1].split("_p")
    return (int(parts[0]), int(parts[1]))

s1_keys = set([get_key(f, "s1") for f in s1_files])
s2_keys = set([get_key(f, "s2") for f in s2_files])
lc_keys = set([get_key(f, "lc") for f in lc_files])

s1_s2_pairs = s1_keys.intersection(s2_keys)
print(f"Valid S1-S2 pairs: {len(s1_s2_pairs)}")

triplets = s1_s2_pairs.intersection(lc_keys)
print(f"Valid S1-S2-LC triplets: {len(triplets)}")

# Inspect a small sample
if len(triplets) > 0:
    scene_id, patch_id = list(triplets)[0]
    s1, s2, lc, bounds = dataset.get_s1s2lc_triplet(
        Seasons.SPRING, scene_id, patch_id, s1_bands=S1Bands.ALL, s2_bands=S2Bands.ALL, lc_bands=LCBands.ALL)
    
    print(f"\nS1 channels: {s1.shape[0]} (expected 2)")
    print(f"S1 stats: min={s1.min():.2f}, max={s1.max():.2f}, mean={s1.mean():.2f}")
    
    print(f"\nS2 channels: {s2.shape[0]} (expected 13)")
    print(f"S2 stats: min={s2.min():.2f}, max={s2.max():.2f}, mean={s2.mean():.2f}")
    
    print(f"\nLC channels loaded: {lc.shape[0]}")
    print(f"LC unique values: {np.unique(lc)}")
    
    # Also load the raw LC file without the dataset loader to see what's actually in there
    lc_raw_path = glob(os.path.join(base_dir, 'ROIs1158_spring', f'lc_{scene_id}', f'*_lc_{scene_id}_p{patch_id}.tif'))[0]
    with rasterio.open(lc_raw_path) as src:
        raw_lc = src.read()
        print(f"Raw LC channels on disk: {raw_lc.shape[0]}")
        print(f"Raw LC unique values on disk: {np.unique(raw_lc)}")

print("\nFinished verification.")
