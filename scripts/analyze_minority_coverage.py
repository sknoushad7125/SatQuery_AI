import os
import sys
import numpy as np
import rasterio
from glob import glob
from multiprocessing import Pool
from collections import defaultdict

# SEN12MS dataset directory
base_dir = os.path.join(os.getcwd(), 'datasets/sen12ms')
lc_files = glob(os.path.join(base_dir, 'ROIs1158_spring', 'lc_*', '*.tif'))

# IGBP -> Project Class Mapping
# 1-12, 14 -> Vegetation (0)
# 13 -> Urban (1)
# 17 -> Water (2)
# 15, 16 -> Bare (3)

def analyze_patch(filepath):
    try:
        with rasterio.open(filepath) as src:
            # Band 1 is IGBP in SEN12MS LCBands.ALL order
            lc = src.read(1)
            
        total = lc.size # 256 * 256 = 65536
        
        # Calculate pixels
        urban = np.sum(lc == 13)
        water = np.sum(lc == 17)
        bare = np.sum((lc == 15) | (lc == 16))
        veg = total - (urban + water + bare)
        
        # Extract Scene ID
        scene_id = int(os.path.basename(filepath).replace(".tif", "").split("lc_")[1].split("_p")[0])
        
        return (filepath, scene_id, urban, water, veg, bare)
    except Exception as e:
        return None

if __name__ == '__main__':
    print(f"Analyzing {len(lc_files)} LC patches...")
    
    with Pool(os.cpu_count()) as p:
        results = p.map(analyze_patch, lc_files)
        
    results = [r for r in results if r is not None]
    
    total_patches = len(results)
    print(f"\nSuccessfully processed {total_patches} patches.")
    
    # Store percentages
    pct_urban = []
    pct_water = []
    pct_veg = []
    pct_bare = []
    
    # Bare counts
    bare_1 = 0
    bare_10 = 0
    bare_01 = 0
    bare_1pct = 0
    bare_5pct = 0
    
    # Urban counts
    urban_1 = 0
    urban_01 = 0
    urban_1pct = 0
    urban_5pct = 0

    # Water counts
    water_1 = 0
    water_01 = 0
    water_1pct = 0
    water_5pct = 0
    
    scene_bare_counts = defaultdict(int)
    scene_bare_pixels = defaultdict(int)

    for r in results:
        filepath, scene_id, urban, water, veg, bare = r
        total = 65536
        
        pu = urban / total * 100
        pw = water / total * 100
        pv = veg / total * 100
        pb = bare / total * 100
        
        pct_urban.append(pu)
        pct_water.append(pw)
        pct_veg.append(pv)
        pct_bare.append(pb)
        
        # Bare thresholds
        if bare >= 1: bare_1 += 1
        if bare >= 10: bare_10 += 1
        if pb >= 0.1: bare_01 += 1
        if pb >= 1.0: bare_1pct += 1
        if pb >= 5.0: bare_5pct += 1
        
        # Urban thresholds
        if urban >= 1: urban_1 += 1
        if pu >= 0.1: urban_01 += 1
        if pu >= 1.0: urban_1pct += 1
        if pu >= 5.0: urban_5pct += 1
        
        # Water thresholds
        if water >= 1: water_1 += 1
        if pw >= 0.1: water_01 += 1
        if pw >= 1.0: water_1pct += 1
        if pw >= 5.0: water_5pct += 1
        
        # Scene stats
        if bare > 0:
            scene_bare_counts[scene_id] += 1
            scene_bare_pixels[scene_id] += bare
            
    pct_bare = np.array(pct_bare)
    
    print("\n### A. Patch coverage table")
    print("Class | Patches with >=1 pixel | >=0.1% | >=1% | >=5%")
    print(f"Urban | {urban_1} | {urban_01} | {urban_1pct} | {urban_5pct}")
    print(f"Water | {water_1} | {water_01} | {water_1pct} | {water_5pct}")
    print(f"Bare  | {bare_1} | {bare_01} | {bare_1pct} | {bare_5pct}")
    
    print(f"\nBare patches with >=10 pixels: {bare_10}")

    print("\n### B. Bare-class statistics")
    print(f"Minimum: {np.min(pct_bare):.4f}%")
    print(f"Median: {np.median(pct_bare):.4f}%")
    print(f"90th percentile: {np.percentile(pct_bare, 90):.4f}%")
    print(f"95th percentile: {np.percentile(pct_bare, 95):.4f}%")
    print(f"99th percentile: {np.percentile(pct_bare, 99):.4f}%")
    print(f"Maximum: {np.max(pct_bare):.4f}%")
    
    print("\n### C. Scene concentration")
    print(f"Total scenes containing at least 1 Bare pixel: {len(scene_bare_counts)}")
    
    # Sort scenes by bare pixel volume
    sorted_scenes = sorted(scene_bare_pixels.items(), key=lambda x: x[1], reverse=True)
    print("\nTop 5 scenes with most Bare pixels:")
    for scene, pixels in sorted_scenes[:5]:
        print(f"Scene {scene}: {pixels} Bare pixels spread across {scene_bare_counts[scene]} patches")
        
    # How concentrated?
    total_bare_pixels = sum(scene_bare_pixels.values())
    if total_bare_pixels > 0:
        top_scene_pct = sorted_scenes[0][1] / total_bare_pixels * 100
        print(f"\nThe #1 scene contains {top_scene_pct:.1f}% of all Bare pixels in the dataset.")
