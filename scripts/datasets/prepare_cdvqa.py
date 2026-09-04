import json
import os
from pathlib import Path

def process_cdvqa():
    base_dir = Path("datasets/cdvqa")
    out_dir = Path("training_data/manifests")
    out_dir.mkdir(parents=True, exist_ok=True)

    # QA files are named Test_*.json, but the instruction says to just use what's there
    # and "create: training_data/manifests/cdvqa_train.jsonl" "If only a train split is safely identifiable, create only the train manifest and document that."
    # Since the files are named Test_, maybe the dataset only shipped with a single split and the authors named it Test.
    questions_file = base_dir / "qa" / "Test_questions.json"
    answers_file = base_dir / "qa" / "Test_answers.json"
    images_file = base_dir / "qa" / "Test_images.json"

    with open(questions_file, 'r') as f:
        q_data = json.load(f)
        questions = q_data.get("questions", q_data) if isinstance(q_data, dict) else q_data

    with open(answers_file, 'r') as f:
        a_data = json.load(f)
        answers = a_data.get("answers", a_data) if isinstance(a_data, dict) else a_data

    with open(images_file, 'r') as f:
        i_data = json.load(f)
        images = i_data.get("images", i_data) if isinstance(i_data, dict) else i_data

    ans_map = {a["question_id"]: a["answer"] for a in answers if a.get("active", True)}

    img_map = {}
    for img in images:
        if img.get("active", True):
            base_name = img["file_name"].replace(".png", "").replace(".tif", "")
            img_map[img["id"]] = base_name

    records = []

    for q in questions:
        if not q.get("active", True):
            continue

        q_id = q["id"]
        img_id = q["img_id"]

        if q_id not in ans_map or img_id not in img_map:
            continue

        base_name = img_map[img_id]

        before = f"datasets/cdvqa/images/{base_name}_1.png"
        after = f"datasets/cdvqa/images/{base_name}_2.png"

        # Verify
        if not os.path.exists(before) or not os.path.exists(after):
            continue

        records.append({
            "dataset": "cdvqa",
            "task": "change_vqa",
            "split": "train",
            "sample_id": f"cdvqa_{q_id}",
            "before_image": before,
            "after_image": after,
            "question": q["question"],
            "answer": ans_map[q_id]
        })

    with open(out_dir / "cdvqa_train.jsonl", 'w') as f:
        for r in records:
            f.write(json.dumps(r) + "\n")

    print(f"CDVQA Prep: Extracted {len(records)} pairs")

if __name__ == "__main__":
    process_cdvqa()
