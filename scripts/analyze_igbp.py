import sys
import os
import rasterio
import numpy as np
from glob import glob

sys.path.append(os.getcwd())
from sen12ms_dataLoader import SEN12MSDataset, Seasons, S1Bands, S2Bands, LCBands

base_dir = os.path.join(os.getcwd(), 'datasets/sen12ms')
dataset = SEN12MSDataset(base_dir)

lc_files = glob(os.path.join(base_dir, 'ROIs1158_spring', 'lc_*', '*.tif'))

# Sample up to 1000 patches for statistical distribution
np.random.seed(42)
sample_files = np.random.choice(lc_files, min(1000, len(lc_files)), replace=False)

counts = np.zeros(256, dtype=np.int64)

for i, f in enumerate(sample_files):
    scene_id = int(os.path.basename(f).replace(".tif", "").split("lc_")[1].split("_p")[0])
    patch_id = int(os.path.basename(f).replace(".tif", "").split("_p")[1])
    try:
        lc, _ = dataset.get_patch(Seasons.SPRING, scene_id, patch_id, LCBands.IGBP)
        patch_counts = np.bincount(lc.flatten(), minlength=256)
        counts += patch_counts
    except Exception:
        pass

total_pixels = counts.sum()

IGBP_NAMES = {
    0: "Water Bodies (or Background)",
    1: "Evergreen Needleleaf Forests",
    2: "Evergreen Broadleaf Forests",
    3: "Deciduous Needleleaf Forests",
    4: "Deciduous Broadleaf Forests",
    5: "Mixed Forests",
    6: "Closed Shrublands",
    7: "Open Shrublands",
    8: "Woody Savannas",
    9: "Savannas",
    10: "Grasslands",
    11: "Permanent Wetlands",
    12: "Croplands",
    13: "Urban and Built-up Lands",
    14: "Cropland/Natural Vegetation Mosaics",
    15: "Permanent Snow and Ice",
    16: "Barren",
    17: "Water Bodies"
}

print("\n=== IGBP Class Distribution (1000-patch sample) ===")
for i in range(256):
    if counts[i] > 0:
        pct = (counts[i] / total_pixels) * 100
        name = IGBP_NAMES.get(i, "Unknown/Invalid")
        print(f"ID {i:<3} | {name:<35} | {counts[i]:<10,d} pixels | {pct:>6.2f}%")

print(f"\nTotal valid pixels: {total_pixels:,d}")
