import os
import json
import glob
from PIL import Image
import numpy as np

def validate_levir(dataset_dir="datasets/levir_cd", report_path="datasets/reports/levir_cd_validation.json"):
    report = {
        "status": "ABSENT",
        "samples": 0,
        "valid": False,
        "details": {}
    }
    
    if not os.path.exists(dataset_dir):
        return report
        
    splits = ["train", "val", "test"]
    total_samples = 0
    errors = []
    
    for split in splits:
        split_dir = os.path.join(dataset_dir, split)
        if not os.path.exists(split_dir):
            continue
            
        a_dir = os.path.join(split_dir, "A")
        b_dir = os.path.join(split_dir, "B")
        l_dir = os.path.join(split_dir, "label")
        
        if not (os.path.exists(a_dir) and os.path.exists(b_dir) and os.path.exists(l_dir)):
            errors.append(f"Missing A, B, or label dir in {split}")
            continue
            
        a_files = sorted(glob.glob(os.path.join(a_dir, "*.png")))
        b_files = sorted(glob.glob(os.path.join(b_dir, "*.png")))
        l_files = sorted(glob.glob(os.path.join(l_dir, "*.png")))
        
        a_names = set([os.path.basename(f) for f in a_files])
        b_names = set([os.path.basename(f) for f in b_files])
        l_names = set([os.path.basename(f) for f in l_files])
        
        if a_names != b_names or a_names != l_names:
            errors.append(f"Filename mismatch in split {split}")
            
        total_samples += len(a_files)
        
        # Check first 3 files for dimensions and mask validity
        for f in a_files[:3]:
            try:
                base = os.path.basename(f)
                img_a = Image.open(f)
                img_b = Image.open(os.path.join(b_dir, base))
                mask = Image.open(os.path.join(l_dir, base))
                
                if img_a.size != img_b.size or img_a.size != mask.size:
                    errors.append(f"Dimension mismatch in {base}")
                
                mask_arr = np.array(mask)
                unique_vals = np.unique(mask_arr)
                if not set(unique_vals).issubset({0, 255, 1}):
                    errors.append(f"Mask {base} has non-binary values: {unique_vals}")
            except Exception as e:
                errors.append(f"Read error on {f}: {e}")

    report["samples"] = total_samples
    if total_samples > 0 and not errors:
        report["status"] = "PRESENT"
        report["valid"] = True
    elif total_samples > 0:
        report["status"] = "PARTIAL/INVALID"
        
    report["details"]["errors"] = errors
    
    os.makedirs(os.path.dirname(report_path), exist_ok=True)
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2)
        
    return report

if __name__ == "__main__":
    print(validate_levir())
