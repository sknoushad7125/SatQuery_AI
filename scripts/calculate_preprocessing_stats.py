import sys
import os
import rasterio
import numpy as np
from glob import glob
import json

sys.path.append(os.getcwd())
from sen12ms_dataLoader import SEN12MSDataset, Seasons, S1Bands, S2Bands, LCBands

base_dir = os.path.join(os.getcwd(), 'datasets/sen12ms')
dataset = SEN12MSDataset(base_dir)

train_scenes = {6, 17, 21, 24, 26, 31, 39, 40, 41, 45, 63, 71, 75, 77, 83, 97, 100, 101, 103, 106, 112, 113, 115, 117, 119, 120, 121, 122, 124, 126, 127, 131, 132, 140, 142, 144, 146, 147, 148}

s1_files_all = glob(os.path.join(base_dir, 'ROIs1158_spring', 's1_*', '*.tif'))

train_files = []
for f in s1_files_all:
    scene_id = int(os.path.basename(f).replace(".tif", "").split("s1_")[1].split("_p")[0])
    if scene_id in train_scenes:
        train_files.append(f)

np.random.seed(42)
sample_files = np.random.choice(train_files, min(1000, len(train_files)), replace=False)

print(f"Sampled {len(sample_files)} TRAIN patches for statistics...")

s1_data = []
s2_data = []

for f in sample_files:
    scene_id = int(os.path.basename(f).replace(".tif", "").split("s1_")[1].split("_p")[0])
    patch_id = int(os.path.basename(f).replace(".tif", "").split("_p")[1])
    try:
        s1, s2, lc, bounds = dataset.get_s1s2lc_triplet(
            Seasons.SPRING, scene_id, patch_id, 
            s1_bands=S1Bands.ALL, s2_bands=S2Bands.ALL, lc_bands=LCBands.IGBP)
        s1_data.append(s1)
        s2_data.append(s2)
    except:
        pass

s1_data = np.stack(s1_data) # [N, 2, 256, 256]
s2_data = np.stack(s2_data) # [N, 13, 256, 256]

s1_stats = {}
s2_stats = {}

print("Calculating S1 stats...")
s1_percentiles = [0.1, 1, 5, 50, 95, 99, 99.9]
for i, band in enumerate(['VV', 'VH']):
    data = s1_data[:, i, :, :]
    pcts = np.percentile(data, s1_percentiles)
    s1_stats[band] = {
        'min': float(np.min(data)),
        'max': float(np.max(data)),
        'p0.1': float(pcts[0]),
        'p1': float(pcts[1]),
        'p5': float(pcts[2]),
        'p50': float(pcts[3]),
        'p95': float(pcts[4]),
        'p99': float(pcts[5]),
        'p99.9': float(pcts[6])
    }

print("Calculating S2 stats...")
s2_percentiles = [1, 50, 99]
for i in range(13):
    data = s2_data[:, i, :, :]
    pcts = np.percentile(data, s2_percentiles)
    s2_stats[f'B{i+1}'] = {
        'min': float(np.min(data)),
        'max': float(np.max(data)),
        'p1': float(pcts[0]),
        'p50': float(pcts[1]),
        'p99': float(pcts[2])
    }

config = {
    'S1': s1_stats,
    'S2': s2_stats
}

with open('scripts/config/optical_sar_preprocessing.json', 'w') as f:
    json.dump(config, f, indent=4)

print("Saved to scripts/config/optical_sar_preprocessing.json")
