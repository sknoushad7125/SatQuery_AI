import os
import json
import hashlib
from datetime import datetime, timezone

def compute_sha256(path):
    if not os.path.exists(path): return None
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()

def verify_rsvqa():
    base_dir = "datasets/rsvqa"
    archives = ["Images_LR.zip", "Questions_LR.zip", "Answers_LR.zip"]
    archive_info = []
    
    for arc in archives:
        arc_path = os.path.join(base_dir, arc)
        if os.path.exists(arc_path):
            archive_info.append({
                "name": arc,
                "present": True,
                "size_bytes": os.path.getsize(arc_path),
                "sha256": compute_sha256(arc_path)
            })
        else:
            archive_info.append({"name": arc, "present": False})
            
    splits = ["train", "val", "test"]
    split_metrics = {}
    
    images_dir = os.path.join(base_dir, "Images_LR")
    actual_images = set(os.listdir(images_dir)) if os.path.exists(images_dir) else set()
    
    missing_refs = []
    all_referenced_imgs = set()
    
    for s in splits:
        metrics = {}
        # Images
        with open(os.path.join(base_dir, f"LR_split_{s}_images.json")) as f:
            imgs = json.load(f)["images"]
            metrics["image_count"] = len(imgs)
            for img in imgs:
                img_filename = f"{img['id']}.tif"
                all_referenced_imgs.add(img_filename)
                if img_filename not in actual_images:
                    missing_refs.append(img_filename)
                    
        # Questions
        with open(os.path.join(base_dir, f"LR_split_{s}_questions.json")) as f:
            qs = json.load(f)["questions"]
            metrics["question_count"] = len(qs)
            if len(qs) > 0:
                metrics["question_schema"] = list(qs[0].keys())
                
        # Answers
        with open(os.path.join(base_dir, f"LR_split_{s}_answers.json")) as f:
            ans = json.load(f)["answers"]
            metrics["answer_count"] = len(ans)
            if len(ans) > 0:
                metrics["answer_schema"] = list(ans[0].keys())
                
        split_metrics[s] = metrics

    # Aggregates
    with open(os.path.join(base_dir, "all_questions.json")) as f:
        all_qs = json.load(f)["questions"]
    with open(os.path.join(base_dir, "all_answers.json")) as f:
        all_ans = json.load(f)["answers"]
        
    sum_qs = sum(m["question_count"] for m in split_metrics.values())
    sum_ans = sum(m["answer_count"] for m in split_metrics.values())
    
    consistency = {
        "all_questions_match": len(all_qs) == sum_qs,
        "all_answers_match": len(all_ans) == sum_ans,
        "missing_image_references": len(missing_refs),
        "unreferenced_image_files": len(actual_images - all_referenced_imgs)
    }

    manifest = {
        "dataset": "RSVQA-LR",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "archives_present": any(a["present"] for a in archive_info),
        "archives": archive_info,
        "splits": split_metrics,
        "consistency": consistency,
        "total_actual_images": len(actual_images)
    }
    
    with open(os.path.join(base_dir, "manifest.json"), "w") as f:
        json.dump(manifest, f, indent=2)
        
    return manifest

def verify_cdvqa():
    base_dir = "datasets/cdvqa/qa"
    
    with open(os.path.join(base_dir, "Test_questions.json")) as f: qs = json.load(f)["questions"]
    with open(os.path.join(base_dir, "Test_answers.json")) as f: ans = json.load(f)["answers"]
    with open(os.path.join(base_dir, "Test_images.json")) as f: imgs = json.load(f)["images"]
    
    unique_image_ids = set(img["id"] for img in imgs)
    unique_filenames = set(img["file_name"] for img in imgs)
    
    manifest = {
        "dataset": "CDVQA",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "metrics": {
            "questions_count": len(qs),
            "answers_count": len(ans),
            "images_referenced": len(imgs),
            "unique_image_ids": len(unique_image_ids),
            "sample_image_filename": imgs[0]["file_name"],
            "matches_levir_format": imgs[0]["file_name"].endswith(".png")
        },
        "verification_status": "CDVQA JSONs verified. Physical mapping to LEVIR-CD pending LEVIR-CD acquisition."
    }
    
    with open("datasets/cdvqa/manifest.json", "w") as f:
        json.dump(manifest, f, indent=2)
        
    return manifest

if __name__ == "__main__":
    r_man = verify_rsvqa()
    c_man = verify_cdvqa()
    
    print("\n=== RSVQA-LR RESULTS ===")
    for a in r_man["archives"]:
        print(f"Archive {a['name']}: {'Present' if a['present'] else 'Absent'}")
    print(f"Total Actual Images: {r_man['total_actual_images']}")
    for s, m in r_man["splits"].items():
        print(f"Split {s}: {m['image_count']} imgs, {m['question_count']} qs, {m['answer_count']} ans")
    print("Consistency:")
    print(f"  All Questions Match: {r_man['consistency']['all_questions_match']}")
    print(f"  All Answers Match: {r_man['consistency']['all_answers_match']}")
    print(f"  Missing Image Refs: {r_man['consistency']['missing_image_references']}")
    print(f"  Unreferenced Imgs: {r_man['consistency']['unreferenced_image_files']}")
    
    print("\n=== CDVQA RESULTS ===")
    print(f"Questions: {c_man['metrics']['questions_count']}")
    print(f"Answers: {c_man['metrics']['answers_count']}")
    print(f"Unique Images: {c_man['metrics']['unique_image_ids']}")
    print(f"Sample Filename: {c_man['metrics']['sample_image_filename']}")
    print(c_man['verification_status'])
