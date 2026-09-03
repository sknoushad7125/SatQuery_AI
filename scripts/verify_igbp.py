import sys
import os
import rasterio
import numpy as np
from glob import glob

sys.path.append(os.getcwd())
from sen12ms_dataLoader import SEN12MSDataset, Seasons, S1Bands, S2Bands, LCBands

base_dir = os.path.join(os.getcwd(), 'datasets/sen12ms')
dataset = SEN12MSDataset(base_dir)

s1, s2, lc, bounds = dataset.get_s1s2lc_triplet(
    Seasons.SPRING, 1, 100, s1_bands=S1Bands.ALL, s2_bands=S2Bands.ALL, lc_bands=LCBands.IGBP)

print(f"IGBP shape: {lc.shape}")
print(f"IGBP unique values: {np.unique(lc)}")
