import os
import rasterio

def get_patch_stratum(mask):
    total_pixels = float(mask.size)
    p_urb = (mask == 1).sum() / total_pixels * 100.0
    p_wat = (mask == 2).sum() / total_pixels * 100.0
    p_bare = (mask == 3).sum() / total_pixels * 100.0
    
    if p_bare >= 1.0:
        return 'bare_rich'
    if p_wat >= 5.0 and p_bare < 1.0:
        return 'water_rich'
    if p_urb >= 5.0 and p_wat < 5.0 and p_bare < 1.0:
        return 'urban_rich'
    if p_urb >= 1.0 or p_wat >= 1.0:
        return 'mixed_minority'
    return 'vegetation_dominant'

def get_train_sampling_weights(dataset):
    targets = {
        'vegetation_dominant': 0.45,
        'urban_rich': 0.25,
        'water_rich': 0.15,
        'mixed_minority': 0.10,
        'bare_rich': 0.05
    }
    
    stratum_counts = {k: 0 for k in targets.keys()}
    patch_strata = []
    
    # Pre-calculate all strata exactly based on actual disk masks
    for idx, (scene_id, patch_id) in enumerate(dataset.samples):
        filename = f"ROIs1158_spring_lc_{scene_id}_p{patch_id}.tif"
        lc_path = os.path.join(dataset.base_dir, 'ROIs1158_spring', f'lc_{scene_id}', filename)
        with rasterio.open(lc_path) as src:
            lc = src.read(1)
        
        mask = dataset.class_map[lc]
        stratum = get_patch_stratum(mask)
        
        patch_strata.append(stratum)
        stratum_counts[stratum] += 1
        
    weights = []
    for stratum in patch_strata:
        # Weight = P(Target) / N(Stratum)
        w = targets[stratum] / stratum_counts[stratum]
        weights.append(w)
        
    return weights, patch_strata, stratum_counts, targets
