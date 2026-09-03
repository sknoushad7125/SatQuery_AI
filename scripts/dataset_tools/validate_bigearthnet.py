import os
import json
import glob

def validate_bigearthnet(dataset_dir="datasets/bigearthnet", report_path="datasets/reports/bigearthnet_validation.json"):
    report = {
        "status": "ABSENT",
        "samples": 0,
        "valid": False,
        "details": {}
    }
    
    if not os.path.exists(dataset_dir):
        return report
        
    img_dir = os.path.join(dataset_dir, "images")
    ann_file = os.path.join(dataset_dir, "annotations.json")
    
    if not os.path.exists(img_dir) or not os.path.exists(ann_file):
        report["details"]["errors"] = ["Missing images directory or annotations.json (BigEarthNet.txt schema)"]
        return report
        
    errors = []
    try:
        with open(ann_file, "r") as f:
            annotations = json.load(f)
            
        valid_samples = 0
        # Check text presence
        for ann in annotations[:10]:
            if "text" not in ann and "caption" not in ann:
                errors.append("Missing text annotations for Image-Text adaptation. Is this BigEarthNet.txt?")
                break
                
        report["samples"] = len(annotations)
        if len(annotations) > 0 and not errors:
            report["status"] = "PRESENT"
            report["valid"] = True
        elif len(annotations) > 0:
            report["status"] = "PARTIAL/INVALID"
            
    except Exception as e:
        errors.append(f"Validation error: {e}")
        
    report["details"]["errors"] = errors
    
    os.makedirs(os.path.dirname(report_path), exist_ok=True)
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2)
        
    return report

if __name__ == "__main__":
    print(validate_bigearthnet())
