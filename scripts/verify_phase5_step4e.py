import sys
import os
import rasterio
import numpy as np

sys.path.append(os.getcwd())
from sen12ms_dataLoader import SEN12MSDataset, Seasons, S1Bands, S2Bands, LCBands

dataset = SEN12MSDataset(os.path.join(os.getcwd(), 'datasets/sen12ms'))
scene_id = 1
# Get patch_ids specifically from s1_1 and s2_1 and find intersection
s1_path = f"datasets/sen12ms/ROIs1158_spring/s1_{scene_id}"
s2_path = f"datasets/sen12ms/ROIs1158_spring/s2_{scene_id}"

s1_patches = [int(p.rsplit("_", 1)[1].split("p")[1].split(".")[0]) for p in os.listdir(s1_path)]
s2_patches = [int(p.rsplit("_", 1)[1].split("p")[1].split(".")[0]) for p in os.listdir(s2_path)]

common_patches = list(set(s1_patches).intersection(set(s2_patches)))
common_patches.sort()

print(f"Checking scene_id {scene_id} which has {len(common_patches)} common patches.")

success_count = 0
for pid in common_patches[:10]:
    try:
        s1, s2, lc, bounds = dataset.get_s1s2lc_triplet(
            Seasons.SPRING, scene_id, pid, s1_bands=S1Bands.ALL, s2_bands=S2Bands.ALL, lc_bands=LCBands.ALL)
        
        if s1.shape[0] != 2:
            print(f"FAIL S1 channels: expected 2, got {s1.shape[0]}")
            continue
        if s2.shape[0] != 13:
            print(f"FAIL S2 channels: expected 13, got {s2.shape[0]}")
            continue
        if lc.shape[0] != 4:
            print(f"FAIL LC channels: expected 4, got {lc.shape[0]}")
            continue
        
        print(f"SUCCESS {pid}: S1={s1.shape}, S2={s2.shape}, LC={lc.shape}")
        success_count += 1
    except Exception as e:
        print(f"FAIL {pid}: {e}")

if success_count == 10:
    print("\n10-sample integrity test: PASS")
else:
    print(f"\n10-sample integrity test: FAIL ({success_count}/10)")
