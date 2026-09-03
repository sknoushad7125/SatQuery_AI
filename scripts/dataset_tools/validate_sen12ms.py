import os
import json
import glob
import rasterio

def validate_sen12ms(dataset_dir="datasets/sen12ms", report_path="datasets/reports/sen12ms_validation.json"):
    report = {
        "status": "ABSENT",
        "samples": 0,
        "valid": False,
        "details": {}
    }
    
    if not os.path.exists(dataset_dir):
        return report
        
    roi_dirs = glob.glob(os.path.join(dataset_dir, "ROIs*"))
    total_triplets = 0
    errors = []
    
    for roi in roi_dirs:
        s1_dirs = glob.glob(os.path.join(roi, "s1_*"))
        s2_dirs = glob.glob(os.path.join(roi, "s2_*"))
        lc_dirs = glob.glob(os.path.join(roi, "lc_*"))
        
        for s1_dir in s1_dirs:
            base_idx = os.path.basename(s1_dir).split("_")[1]
            s2_dir = os.path.join(roi, f"s2_{base_idx}")
            lc_dir = os.path.join(roi, f"lc_{base_idx}")
            
            if not os.path.exists(s2_dir) or not os.path.exists(lc_dir):
                errors.append(f"Missing s2 or lc dir for {s1_dir}")
                continue
                
            s1_files = sorted(glob.glob(os.path.join(s1_dir, "*.tif")))
            for s1_f in s1_files:
                base_name = os.path.basename(s1_f)
                s2_f = os.path.join(s2_dir, base_name.replace("s1", "s2"))
                lc_f = os.path.join(lc_dir, base_name.replace("s1", "lc"))
                
                if os.path.exists(s2_f) and os.path.exists(lc_f):
                    total_triplets += 1
                else:
                    errors.append(f"Missing matched files for {base_name}")
                    continue
                    
                if total_triplets <= 3:
                    try:
                        with rasterio.open(s1_f) as src:
                            if src.count != 2: errors.append(f"SAR {base_name} has {src.count} channels, expected 2 (VV/VH)")
                        with rasterio.open(s2_f) as src:
                            if src.count not in [13, 10, 4, 3]: errors.append(f"Optical {base_name} unexpected channels: {src.count}")
                    except Exception as e:
                        errors.append(f"Read error on {base_name}: {e}")
                        
    report["samples"] = total_triplets
    if total_triplets > 0 and not errors:
        report["status"] = "PRESENT"
        report["valid"] = True
    elif total_triplets > 0:
        report["status"] = "PARTIAL/INVALID"
        
    report["details"]["errors"] = errors
    
    os.makedirs(os.path.dirname(report_path), exist_ok=True)
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2)
        
    return report

if __name__ == "__main__":
    print(validate_sen12ms())
