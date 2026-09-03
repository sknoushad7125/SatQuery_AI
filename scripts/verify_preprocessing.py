import sys
import os
import rasterio
import numpy as np
from glob import glob

sys.path.append(os.getcwd())
from sen12ms_dataLoader import SEN12MSDataset, Seasons, S1Bands, S2Bands, LCBands

base_dir = os.path.join(os.getcwd(), 'datasets/sen12ms')
dataset = SEN12MSDataset(base_dir)

# 1. Inspect Loader Enums
print("--- S1 Bands ---")
for b in S1Bands: print(f"{b.name} = {b.value}")
print("\n--- S2 Bands ---")
for b in S2Bands: print(f"{b.name} = {b.value}")

# 2. Sample 100 patches for stats
s1_files = glob(os.path.join(base_dir, 'ROIs1158_spring', 's1_*', '*.tif'))
np.random.seed(42)
sample_files = np.random.choice(s1_files, min(100, len(s1_files)), replace=False)

s1_data = []
s2_data = []

print("\nProcessing 100 patches to compute statistics...")
for f in sample_files:
    scene_id = int(os.path.basename(f).replace(".tif", "").split("s1_")[1].split("_p")[0])
    patch_id = int(os.path.basename(f).replace(".tif", "").split("_p")[1])
    
    try:
        s1, s2, lc, bounds = dataset.get_s1s2lc_triplet(
            Seasons.SPRING, scene_id, patch_id, 
            s1_bands=S1Bands.ALL, s2_bands=S2Bands.ALL, lc_bands=LCBands.IGBP)
        s1_data.append(s1)
        s2_data.append(s2)
    except Exception as e:
        continue

s1_data = np.stack(s1_data) # [100, 2, 256, 256]
s2_data = np.stack(s2_data) # [100, 13, 256, 256]

print("\n--- Sentinel-1 Statistics ---")
print(f"Shape: {s1_data.shape}")
print(f"Min: {np.min(s1_data):.2f}, Max: {np.max(s1_data):.2f}, Mean: {np.mean(s1_data):.2f}")
pct = [1, 5, 50, 95, 99]
for i in range(2): # VV, VH
    band_name = "VV" if i == 0 else "VH"
    band_data = s1_data[:, i, :, :]
    vals = np.percentile(band_data, pct)
    print(f"Band {band_name}: Min={np.min(band_data):.2f}, Max={np.max(band_data):.2f}")
    print(f"Percentiles (1, 5, 50, 95, 99): {['{:.2f}'.format(v) for v in vals]}")

print("\n--- Sentinel-2 Statistics ---")
print(f"Shape: {s2_data.shape}")
print(f"Min: {np.min(s2_data):.2f}, Max: {np.max(s2_data):.2f}, Mean: {np.mean(s2_data):.2f}")
for i in range(13):
    band_data = s2_data[:, i, :, :]
    vals = np.percentile(band_data, pct)
    print(f"Band {i+1} Percentiles (1, 5, 50, 95, 99): {['{:.2f}'.format(v) for v in vals]}")

print("\n--- Ground Truth Mapping Check ---")
igbp_to_project = {
    1: 0, 2: 0, 3: 0, 4: 0, 5: 0, 6: 0, 7: 0, 8: 0, 9: 0, 10: 0, 11: 0, 12: 0, 14: 0, # Vegetation (0)
    13: 1, # Urban (1)
    17: 2, # Water (2)
    15: 3, 16: 3 # Bare (3)
}
test_values = set(range(1, 18))
mapped_values = set(igbp_to_project[v] for v in test_values)
print(f"All 1-17 IGBP classes accounted for in map keys: {all(k in igbp_to_project for k in range(1, 18))}")
print(f"Resulting unique target classes: {mapped_values} (Expected: {{0, 1, 2, 3}})")

# Verify Spatial Co-registration Metadata
# sen12ms dataset doesn't store spatial info in standard gdal if read as numpy, 
# but bounds are returned by get_patch()
print("\n--- Co-registration Metadata Check ---")
try:
    s1, bounds1 = dataset.get_patch(Seasons.SPRING, scene_id, patch_id, S1Bands.ALL)
    s2, bounds2 = dataset.get_patch(Seasons.SPRING, scene_id, patch_id, S2Bands.ALL)
    lc, bounds_lc = dataset.get_patch(Seasons.SPRING, scene_id, patch_id, LCBands.IGBP)
    print(f"S1 Bounds: {bounds1}")
    print(f"S2 Bounds: {bounds2}")
    print(f"LC Bounds: {bounds_lc}")
    print(f"Bounds exactly match? {bounds1 == bounds2 == bounds_lc}")
except Exception as e:
    print(f"Error checking bounds: {e}")

