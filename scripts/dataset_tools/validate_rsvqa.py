import os
import json
import glob

def validate_rsvqa(dataset_dir="datasets/rsvqa", report_path="datasets/reports/rsvqa_validation.json"):
    report = {
        "status": "ABSENT",
        "samples": 0,
        "valid": False,
        "details": {}
    }
    
    if not os.path.exists(dataset_dir):
        return report
        
    img_dir = os.path.join(dataset_dir, "Images_LR")
    q_file = os.path.join(dataset_dir, "LR_questions.json")
    a_file = os.path.join(dataset_dir, "LR_answers.json")
    
    if not os.path.exists(img_dir) or not os.path.exists(q_file):
        report["details"]["errors"] = ["Missing Images_LR directory or questions JSON"]
        return report
        
    errors = []
    try:
        with open(q_file, "r") as f:
            questions = json.load(f)
        
        answers = []
        if os.path.exists(a_file):
            with open(a_file, "r") as f:
                answers = json.load(f)
                
        img_files = set(glob.glob(os.path.join(img_dir, "*.tif")))
        
        valid_q = 0
        for q in questions[:100]: # Sample check
            img_id = q.get("img_id", q.get("image_id"))
            img_path = os.path.join(img_dir, f"{img_id}.tif")
            if img_path in img_files:
                valid_q += 1
            else:
                errors.append(f"Missing image {img_id}.tif for question {q.get('id')}")
                
        report["samples"] = len(questions)
        if len(questions) > 0 and not errors:
            report["status"] = "PRESENT"
            report["valid"] = True
        elif len(questions) > 0:
            report["status"] = "PARTIAL/INVALID"
            
    except Exception as e:
        errors.append(f"Validation error: {e}")
        
    report["details"]["errors"] = errors
    
    os.makedirs(os.path.dirname(report_path), exist_ok=True)
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2)
        
    return report

if __name__ == "__main__":
    print(validate_rsvqa())
