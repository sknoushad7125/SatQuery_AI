import json
import os
import random
from pathlib import Path

def process_rsvqa():
    base_dir = Path("datasets/rsvqa")
    out_dir = Path("training_data/manifests")
    out_dir.mkdir(parents=True, exist_ok=True)

    questions_file = base_dir / "LR_split_train_questions.json"
    answers_file = base_dir / "LR_split_train_answers.json"
    images_file = base_dir / "LR_split_train_images.json"

    # Load JSONs. Typically RSVQA JSONs are dicts containing {"questions": [...]}, {"answers": [...]}, {"images": [...]}
    with open(questions_file, 'r') as f:
        q_data = json.load(f)
        questions = q_data.get("questions", q_data) if isinstance(q_data, dict) else q_data

    with open(answers_file, 'r') as f:
        a_data = json.load(f)
        answers = a_data.get("answers", a_data) if isinstance(a_data, dict) else a_data

    with open(images_file, 'r') as f:
        i_data = json.load(f)
        images = i_data.get("images", i_data) if isinstance(i_data, dict) else i_data

    # Build mappings
    # Images: id -> original_name / filename
    img_map = {}
    for img in images:
        if isinstance(img, dict) and "id" in img and "active" in img:
            if img["active"]:
                # RSVQA LR uses .tif extension
                name = f"{img['id']}.tif"
                if not name.endswith(".tif"):
                    name += ".tif"
                img_map[img["id"]] = name

    # Answers: question_id -> answer_string
    ans_map = {}
    for ans in answers:
        if isinstance(ans, dict) and "question_id" in ans and "active" in ans:
            if ans["active"]:
                ans_map[ans["question_id"]] = ans["answer"]

    records_by_image = {}
    total_valid = 0
    missing_image_count = 0

    for q in questions:
        if not isinstance(q, dict) or not q.get("active"):
            continue

        q_id = q["id"]
        img_id = q["img_id"]
        question_text = q["question"]

        if q_id not in ans_map:
            continue

        if img_id not in img_map:
            missing_image_count += 1
            continue

        img_filename = img_map[img_id]
        img_path = f"datasets/rsvqa/Images_LR/{img_filename}"

        # Check if file exists locally
        if not os.path.exists(img_path):
            missing_image_count += 1
            continue

        ans_text = ans_map[q_id]

        record = {
            "dataset": "rsvqa",
            "task": "vqa",
            "split": "train",
            "sample_id": f"rsvqa_{q_id}",
            "image": img_path,
            "question": question_text,
            "answer": ans_text,
            "original_split": "train",
            "source_id": q_id
        }

        if img_path not in records_by_image:
            records_by_image[img_path] = []
        records_by_image[img_path].append(record)
        total_valid += 1

    # Split by image
    all_image_paths = sorted(list(records_by_image.keys()))
    random.seed(42)
    random.shuffle(all_image_paths)

    val_size = int(len(all_image_paths) * 0.1)
    val_images = set(all_image_paths[:val_size])

    train_records = []
    val_records = []

    for img_path, recs in records_by_image.items():
        if img_path in val_images:
            for r in recs:
                r["split"] = "internal_val"
                val_records.append(r)
        else:
            for r in recs:
                r["split"] = "train"
                train_records.append(r)

    # Write output
    with open(out_dir / "rsvqa_train.jsonl", 'w') as f:
        for r in train_records:
            f.write(json.dumps(r) + "\n")

    with open(out_dir / "rsvqa_internal_val.jsonl", 'w') as f:
        for r in val_records:
            f.write(json.dumps(r) + "\n")

    print(f"RSVQA Prep: Found {len(train_records)} train records, {len(val_records)} internal_val records. Missing images for {missing_image_count} records.")

if __name__ == "__main__":
    process_rsvqa()
