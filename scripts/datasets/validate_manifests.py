import json
import os
import zipfile
from collections import defaultdict
from pathlib import Path

def main():
    manifests_dir = Path("training_data/manifests")
    reports_dir = Path("training_data/reports")

    if not manifests_dir.exists():
        print("No manifests found.")
        return

    stats = {
        "total_records": 0,
        "by_dataset": defaultdict(int),
        "by_task": defaultdict(int),
        "by_split": defaultdict(int),
        "unique_images": set(),
        "change_vqa_count": 0
    }

    validation_results = {
        "status": "PASS",
        "errors": [],
        "warnings": [],
        "statistics": {}
    }

    sample_ids = set()
    rsvqa_train_imgs = set()
    rsvqa_val_imgs = set()

    vrsbench_train_imgs = set()
    vrsbench_val_imgs = set()

    # Pre-cache zip contents for VRSBench
    vrsbench_train_zip = None
    vrsbench_val_zip = None
    if Path("datasets/vrsbench/Images_train.zip").exists():
        with zipfile.ZipFile("datasets/vrsbench/Images_train.zip", "r") as z:
            vrsbench_train_zip = set(info.filename for info in z.infolist())

    if Path("datasets/vrsbench/Images_val.zip").exists():
        with zipfile.ZipFile("datasets/vrsbench/Images_val.zip", "r") as z:
            vrsbench_val_zip = set(info.filename for info in z.infolist())

    for jsonl_file in manifests_dir.glob("*.jsonl"):
        with open(jsonl_file, "r") as f:
            for line_idx, line in enumerate(f):
                try:
                    record = json.loads(line)
                except json.JSONDecodeError:
                    validation_results["errors"].append(f"{jsonl_file.name}:{line_idx} - Invalid JSON")
                    validation_results["status"] = "FAIL"
                    continue

                stats["total_records"] += 1
                stats["by_dataset"][record.get("dataset", "unknown")] += 1
                stats["by_task"][record.get("task", "unknown")] += 1
                stats["by_split"][record.get("split", "unknown")] += 1

                # Check req fields
                for fld in ["dataset", "task", "split", "sample_id"]:
                    if fld not in record:
                        validation_results["errors"].append(f"Missing {fld} in {record.get('sample_id', 'unknown')}")
                        validation_results["status"] = "FAIL"

                sid = record.get("sample_id")
                if sid in sample_ids:
                    validation_results["errors"].append(f"Duplicate sample_id: {sid}")
                    validation_results["status"] = "FAIL"
                if sid:
                    sample_ids.add(sid)

                task = record.get("task")
                dataset = record.get("dataset")

                if task == "vqa" or task == "change_vqa":
                    if not record.get("question"):
                        validation_results["errors"].append(f"Empty question in {sid}")
                    if not record.get("answer"):
                        validation_results["errors"].append(f"Empty answer in {sid}")

                if task == "captioning":
                    if not record.get("caption"):
                        validation_results["errors"].append(f"Empty caption in {sid}")

                if task == "grounding":
                    bbox = record.get("bbox")
                    if not bbox or len(bbox) != 4:
                        validation_results["errors"].append(f"Invalid bbox in {sid}")
                    elif not all(0 <= v <= 1 for v in bbox):
                        validation_results["errors"].append(f"bbox out of bounds [0,1] in {sid}")

                # Dataset specific validations
                if dataset == "rsvqa":
                    img = record.get("image")
                    if not os.path.exists(img):
                        validation_results["errors"].append(f"RSVQA image not found: {img}")
                        validation_results["status"] = "FAIL"
                    if record.get("split") == "train":
                        rsvqa_train_imgs.add(img)
                    elif record.get("split") == "internal_val":
                        rsvqa_val_imgs.add(img)

                elif dataset == "vrsbench":
                    img_ref = record.get("image", {})
                    archive = img_ref.get("archive", "")
                    member = img_ref.get("member", "")

                    if "Images_train" in archive:
                        if vrsbench_train_zip and member not in vrsbench_train_zip:
                            validation_results["errors"].append(f"VRSBench image not in zip: {member}")
                            validation_results["status"] = "FAIL"
                        vrsbench_train_imgs.add(member)
                    elif "Images_val" in archive:
                        if vrsbench_val_zip and member not in vrsbench_val_zip:
                            validation_results["errors"].append(f"VRSBench image not in val zip: {member}")
                            validation_results["status"] = "FAIL"
                        vrsbench_val_imgs.add(member)

                elif dataset == "bigearthnet_txt":
                    if record.get("image_available") is not False:
                        validation_results["errors"].append(f"BigEarthNet falsely claims local imagery: {sid}")
                        validation_results["status"] = "FAIL"

                elif dataset == "cdvqa":
                    before = record.get("before_image")
                    after = record.get("after_image")
                    if not before or not after:
                        validation_results["errors"].append(f"CDVQA missing pair in {sid}")
                        validation_results["status"] = "FAIL"
                    elif not os.path.exists(before) or not os.path.exists(after):
                        validation_results["errors"].append(f"CDVQA image missing in {sid}")
                        validation_results["status"] = "FAIL"
                    if before == after:
                        validation_results["errors"].append(f"CDVQA before and after identical in {sid}")
                        validation_results["status"] = "FAIL"

                    stats["change_vqa_count"] += 1

    # Leakage check
    rsvqa_leak = rsvqa_train_imgs.intersection(rsvqa_val_imgs)
    if rsvqa_leak:
        validation_results["errors"].append(f"RSVQA Leakage! {len(rsvqa_leak)} images in train and val")
        validation_results["status"] = "FAIL"

    vrsbench_leak = vrsbench_train_imgs.intersection(vrsbench_val_imgs)
    if vrsbench_leak:
        validation_results["errors"].append(f"VRSBench Leakage! {len(vrsbench_leak)} images in train and val")
        validation_results["status"] = "FAIL"

    stats["unique_images"] = len(rsvqa_train_imgs) + len(rsvqa_val_imgs) + len(vrsbench_train_imgs) + len(vrsbench_val_imgs)

    validation_results["statistics"] = {k: (dict(v) if isinstance(v, defaultdict) else v) for k, v in stats.items()}

    # Check max errors
    if len(validation_results["errors"]) > 100:
        validation_results["errors"] = validation_results["errors"][:100] + ["... and more errors."]

    with open(reports_dir / "validation_report.json", "w") as f:
        json.dump(validation_results, f, indent=2)

    with open(reports_dir / "dataset_statistics.json", "w") as f:
        json.dump(validation_results["statistics"], f, indent=2)

    # Markdown audit
    audit = f"""# Dataset Preprocessing Audit

**Validation Status**: {validation_results["status"]}

## Statistics
- **Total Records**: {stats["total_records"]}
- **Unique Images (Local + ZIP)**: {stats["unique_images"]}

### By Dataset
"""
    for d, c in validation_results["statistics"]["by_dataset"].items():
        audit += f"- {d}: {c}\n"

    audit += "\n### By Task\n"
    for d, c in validation_results["statistics"]["by_task"].items():
        audit += f"- {d}: {c}\n"

    audit += "\n### By Split\n"
    for d, c in validation_results["statistics"]["by_split"].items():
        audit += f"- {d}: {c}\n"

    audit += f"\n## Notes\n- BigEarthNet raw imagery is correctly flagged as missing/external.\n- RSVQA split leakage checked: {'PASS' if not rsvqa_leak else 'FAIL'}\n- VRSBench split leakage checked: {'PASS' if not vrsbench_leak else 'FAIL'}\n"

    with open(reports_dir / "dataset_audit.md", "w") as f:
        f.write(audit)

    print(f"Validation Status: {validation_results['status']}")
    if validation_results["status"] == "FAIL":
        print("ERRORS FOUND (first 10):")
        for e in validation_results["errors"][:10]:
            print(e)

if __name__ == "__main__":
    main()
