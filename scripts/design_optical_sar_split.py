import os
import sys
import numpy as np
import rasterio
from glob import glob
from collections import defaultdict
from multiprocessing import Pool
import random

base_dir = os.path.join(os.getcwd(), 'datasets/sen12ms')
lc_files = glob(os.path.join(base_dir, 'ROIs1158_spring', 'lc_*', '*.tif'))

def analyze_patch(filepath):
    try:
        with rasterio.open(filepath) as src:
            lc = src.read(1)
        total = lc.size
        urban = np.sum(lc == 13)
        water = np.sum(lc == 17)
        bare = np.sum((lc == 15) | (lc == 16))
        veg = total - (urban + water + bare)
        scene_id = int(os.path.basename(filepath).replace(".tif", "").split("lc_")[1].split("_p")[0])
        return (scene_id, urban, water, veg, bare)
    except:
        return None

if __name__ == '__main__':
    print("Gathering scene statistics...")
    with Pool(os.cpu_count()) as p:
        results = p.map(analyze_patch, lc_files)
        
    results = [r for r in results if r is not None]
    
    # Aggregate per scene
    scene_stats = defaultdict(lambda: {
        'patches': 0, 'urban': 0, 'water': 0, 'veg': 0, 'bare': 0,
        'bare_1pct': 0, 'bare_5pct': 0, 'urban_any': 0, 'water_any': 0
    })
    
    for r in results:
        scene_id, u, w, v, b = r
        st = scene_stats[scene_id]
        st['patches'] += 1
        st['urban'] += u
        st['water'] += w
        st['veg'] += v
        st['bare'] += b
        total = u + w + v + b
        
        if (b / total) >= 0.01: st['bare_1pct'] += 1
        if (b / total) >= 0.05: st['bare_5pct'] += 1
        if u > 0: st['urban_any'] += 1
        if w > 0: st['water_any'] += 1
        
    scenes = list(scene_stats.keys())
    
    # Try finding a good split
    best_split = None
    best_score = float('inf')
    
    # Target proportions: Train 70%, Val 15%, Test 15%
    for i in range(10000):
        random.seed(i)
        shuffled = scenes.copy()
        random.shuffle(shuffled)
        
        n_scenes = len(shuffled)
        train_scenes = set(shuffled[:int(0.7 * n_scenes)])
        val_scenes = set(shuffled[int(0.7 * n_scenes):int(0.85 * n_scenes)])
        test_scenes = set(shuffled[int(0.85 * n_scenes):])
        
        # Calculate Bare patches with >=5% for each split
        bare5_train = sum(scene_stats[s]['bare_5pct'] for s in train_scenes)
        bare5_val = sum(scene_stats[s]['bare_5pct'] for s in val_scenes)
        bare5_test = sum(scene_stats[s]['bare_5pct'] for s in test_scenes)
        
        # We want >= 15 patches with >=5% Bare in Val and Test to ensure meaningful evaluation
        if bare5_val < 15 or bare5_test < 15:
            continue
            
        # We also want overall patch count to be roughly 70/15/15
        p_train = sum(scene_stats[s]['patches'] for s in train_scenes)
        p_val = sum(scene_stats[s]['patches'] for s in val_scenes)
        p_test = sum(scene_stats[s]['patches'] for s in test_scenes)
        total_p = p_train + p_val + p_test
        
        r_train, r_val, r_test = p_train/total_p, p_val/total_p, p_test/total_p
        
        # Penalty for deviating from 70/15/15
        score = abs(r_train - 0.70) + abs(r_val - 0.15) + abs(r_test - 0.15)
        
        if score < best_score:
            best_score = score
            best_split = (train_scenes, val_scenes, test_scenes)
            
    if best_split is None:
        print("Failed to find a suitable split. Run again or relax constraints.")
        sys.exit(1)
        
    train_scenes, val_scenes, test_scenes = best_split
    
    print("\n### FINAL PROPOSED SPLIT ###")
    print(f"Train Scenes: {sorted(list(train_scenes))}")
    print(f"Val Scenes: {sorted(list(val_scenes))}")
    print(f"Test Scenes: {sorted(list(test_scenes))}")
    
    def print_split_stats(name, split_scenes):
        patches = sum(scene_stats[s]['patches'] for s in split_scenes)
        u = sum(scene_stats[s]['urban'] for s in split_scenes)
        w = sum(scene_stats[s]['water'] for s in split_scenes)
        v = sum(scene_stats[s]['veg'] for s in split_scenes)
        b = sum(scene_stats[s]['bare'] for s in split_scenes)
        total_px = u + w + v + b
        
        bare1 = sum(scene_stats[s]['bare_1pct'] for s in split_scenes)
        bare5 = sum(scene_stats[s]['bare_5pct'] for s in split_scenes)
        u_any = sum(scene_stats[s]['urban_any'] for s in split_scenes)
        w_any = sum(scene_stats[s]['water_any'] for s in split_scenes)
        
        print(f"\n--- {name} Split ---")
        print(f"Scenes: {len(split_scenes)}")
        print(f"Patches: {patches} ({patches/len(results)*100:.1f}%)")
        print(f"Pixels - Veg: {v} ({v/total_px*100:.2f}%)")
        print(f"Pixels - Urban: {u} ({u/total_px*100:.2f}%)")
        print(f"Pixels - Water: {w} ({w/total_px*100:.2f}%)")
        print(f"Pixels - Bare: {b} ({b/total_px*100:.4f}%)")
        print(f"Patches with >=1% Bare: {bare1}")
        print(f"Patches with >=5% Bare: {bare5}")
        print(f"Patches containing Urban: {u_any}")
        print(f"Patches containing Water: {w_any}")
        
    print_split_stats("Train", train_scenes)
    print_split_stats("Validation", val_scenes)
    print_split_stats("Test", test_scenes)
    
    print("\n### LEAKAGE CHECK ###")
    print(f"Train ∩ Val = {train_scenes.intersection(val_scenes)}")
    print(f"Train ∩ Test = {train_scenes.intersection(test_scenes)}")
    print(f"Val ∩ Test = {val_scenes.intersection(test_scenes)}")
