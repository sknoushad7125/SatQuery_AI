import sys
import os
from glob import glob

base_dir = os.path.join(os.getcwd(), 'datasets/sen12ms')

seasons = ['ROIs1158_spring', 'ROIs1868_summer', 'ROIs1970_fall', 'ROIs2017_winter']

for season in seasons:
    s_path = os.path.join(base_dir, season)
    if os.path.exists(s_path):
        s1 = len(glob(os.path.join(s_path, 's1_*', '*.tif')))
        s2 = len(glob(os.path.join(s_path, 's2_*', '*.tif')))
        lc = len(glob(os.path.join(s_path, 'lc_*', '*.tif')))
        print(f"Season: {season} -> S1: {s1}, S2: {s2}, LC: {lc}")
    else:
        print(f"Season: {season} -> NOT FOUND")
