import json
import zipfile
import re
from pathlib import Path

def parse_bbox(bbox_str):
    # Match format like {<25><40><33><60>}
    matches = re.findall(r'<(\d+)>', bbox_str)
    if len(matches) == 4:
        ymin, xmin, ymax, xmax = [int(m)/100.0 for m in matches]
        return [xmin, ymin, xmax, ymax]
    return None

def process_vrsbench():
    base_dir = Path("datasets/vrsbench")
    out_dir = Path("training_data/manifests")
    out_dir.mkdir(parents=True, exist_ok=True)

    train_json_path = base_dir / "VRSBench_train.json"

    # 1. Inspect training zip
    train_zip_path = base_dir / "Images_train.zip"
    valid_train_images = set()
    with zipfile.ZipFile(train_zip_path, 'r') as z:
        for info in z.infolist():
            if not info.filename.startswith("__MACOSX") and not info.filename.endswith("/"):
                # info.filename looks like Images_train/00002_0000.png
                valid_train_images.add(info.filename.split("/")[-1])

    # Parse training
    with open(train_json_path, 'r') as f:
        train_data = json.load(f)

    records_vqa = []
    records_cap = []
    records_ref = []

    sample_id_counter = 0
    for item in train_data:
        img_filename = item["image"]
        if img_filename not in valid_train_images:
            continue

        convs = item.get("conversations", [])
        if len(convs) < 2:
            continue

        q_text = convs[0]["value"]
        a_text = convs[1]["value"]

        img_ref = {
            "storage": "zip",
            "archive": "datasets/vrsbench/Images_train.zip",
            "member": f"Images_train/{img_filename}"
        }

        if "[vqa]" in q_text:
            clean_q = q_text.replace("<image>\n", "").replace("[vqa] ", "").strip()
            records_vqa.append({
                "dataset": "vrsbench",
                "task": "vqa",
                "split": "train",
                "sample_id": f"vrsbench_train_vqa_{sample_id_counter}",
                "image": img_ref,
                "question": clean_q,
                "answer": a_text
            })

        elif "[caption]" in q_text:
            clean_q = q_text.replace("<image>\n", "").replace("[caption] ", "").strip()
            records_cap.append({
                "dataset": "vrsbench",
                "task": "captioning",
                "split": "train",
                "sample_id": f"vrsbench_train_cap_{sample_id_counter}",
                "image": img_ref,
                "caption": a_text
            })

        elif "[refer]" in q_text:
            clean_q = q_text.replace("<image>\n", "").replace("[refer] ", "").strip()
            # Extract expression
            # e.g., "tell me the location for <p>The toll station...</p>?"
            p_match = re.search(r'<p>(.*?)</p>', clean_q)
            text_exp = p_match.group(1) if p_match else clean_q

            bbox = parse_bbox(a_text)
            if bbox:
                # validate 0..1
                if all(0 <= v <= 1 for v in bbox):
                    records_ref.append({
                        "dataset": "vrsbench",
                        "task": "grounding",
                        "split": "train",
                        "sample_id": f"vrsbench_train_ref_{sample_id_counter}",
                        "image": img_ref,
                        "text": text_exp,
                        "bbox": bbox
                    })
        sample_id_counter += 1

    # Write train
    def write_jsonl(path, records):
        with open(path, 'w') as f:
            for r in records:
                f.write(json.dumps(r) + "\n")

    write_jsonl(out_dir / "vrsbench_vqa_train.jsonl", records_vqa)
    write_jsonl(out_dir / "vrsbench_caption_train.jsonl", records_cap)
    write_jsonl(out_dir / "vrsbench_grounding_train.jsonl", records_ref)

    # 2. Validation
    val_zip_path = base_dir / "Images_val.zip"
    valid_val_images = set()
    with zipfile.ZipFile(val_zip_path, 'r') as z:
        for info in z.infolist():
            if not info.filename.startswith("__MACOSX") and not info.filename.endswith("/"):
                valid_val_images.add(info.filename.split("/")[-1])

    # Val files are directly available in vrsbench
    # VRSBench_EVAL_Cap.json, VRSBench_EVAL_referring.json, VRSBench_EVAL_vqa.json
    records_val_vqa = []
    records_val_cap = []
    records_val_ref = []

    val_vqa_path = base_dir / "VRSBench_EVAL_vqa.json"
    if val_vqa_path.exists():
        with open(val_vqa_path, 'r') as f:
            vqa_data = json.load(f)
            for item in vqa_data:
                img = item.get("image_id", "")
                if img in valid_val_images:
                    img_ref = {
                        "storage": "zip",
                        "archive": "datasets/vrsbench/Images_val.zip",
                        "member": f"Images_val/{img}"
                    }
                    records_val_vqa.append({
                        "dataset": "vrsbench",
                        "task": "vqa",
                        "split": "val",
                        "sample_id": f"vrsbench_val_vqa_{item.get('question_id', len(records_val_vqa))}",
                        "image": img_ref,
                        "question": item.get("question"),
                        "answer": item.get("answer")
                    })

    val_cap_path = base_dir / "VRSBench_EVAL_Cap.json"
    if val_cap_path.exists():
        with open(val_cap_path, 'r') as f:
            cap_data = json.load(f)
            if isinstance(cap_data, list):
                for item in cap_data:
                    img = item.get("image_id", "")
                    if img in valid_val_images:
                        img_ref = {
                            "storage": "zip",
                            "archive": "datasets/vrsbench/Images_val.zip",
                            "member": f"Images_val/{img}"
                        }
                        records_val_cap.append({
                            "dataset": "vrsbench",
                            "task": "captioning",
                            "split": "val",
                            "sample_id": f"vrsbench_val_cap_{item.get('question_id', len(records_val_cap))}",
                            "image": img_ref,
                            "caption": item.get("ground_truth")
                        })

    val_ref_path = base_dir / "VRSBench_EVAL_referring.json"
    if val_ref_path.exists():
        with open(val_ref_path, 'r') as f:
            ref_data = json.load(f)
            for item in ref_data:
                img = item.get("image_id", "")
                if img in valid_val_images:
                    img_ref = {
                        "storage": "zip",
                        "archive": "datasets/vrsbench/Images_val.zip",
                        "member": f"Images_val/{img}"
                    }
                    bbox = parse_bbox(item.get("ground_truth", ""))
                    if bbox and all(0 <= v <= 1 for v in bbox):
                        records_val_ref.append({
                            "dataset": "vrsbench",
                            "task": "grounding",
                            "split": "val",
                            "sample_id": f"vrsbench_val_ref_{item.get('question_id', len(records_val_ref))}",
                            "image": img_ref,
                            "text": item.get("question"),
                            "bbox": bbox
                        })

    write_jsonl(out_dir / "vrsbench_vqa_val.jsonl", records_val_vqa)
    write_jsonl(out_dir / "vrsbench_caption_val.jsonl", records_val_cap)
    write_jsonl(out_dir / "vrsbench_grounding_val.jsonl", records_val_ref)

    print(f"VRSBench Prep: Extracted {len(records_vqa)} train VQA, {len(records_cap)} train CAP, {len(records_ref)} train REF")
    print(f"VRSBench Prep: Extracted {len(records_val_vqa)} val VQA, {len(records_val_cap)} val CAP, {len(records_val_ref)} val REF")

if __name__ == "__main__":
    process_vrsbench()
