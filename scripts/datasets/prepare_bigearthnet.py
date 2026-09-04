import json
import re
from pathlib import Path
import pyarrow.parquet as pq

def parse_ben_bbox(bbox_str):
    try:
        if not bbox_str: return None
        clean = bbox_str.replace('[', '').replace(']', '').replace(',', ' ')
        parts = clean.split()
        if len(parts) == 4:
            bbox = [float(p) for p in parts]
            if all(0 <= v <= 1 for v in bbox):
                return bbox
    except Exception:
        pass
    return None

def process_bigearthnet():
    parquet_path = "datasets/bigearthnet_txt/BigEarthNet.txt.parquet"
    out_dir = Path("training_data/manifests")
    out_dir.mkdir(parents=True, exist_ok=True)

    files = {
        "vqa_train": open(out_dir / "bigearthnet_vqa_train.jsonl", "w"),
        "vqa_val": open(out_dir / "bigearthnet_vqa_val.jsonl", "w"),
        "caption_train": open(out_dir / "bigearthnet_caption_train.jsonl", "w"),
        "caption_val": open(out_dir / "bigearthnet_caption_val.jsonl", "w"),
        "grounding_train": open(out_dir / "bigearthnet_grounding_train.jsonl", "w"),
        "grounding_val": open(out_dir / "bigearthnet_grounding_val.jsonl", "w"),
    }

    counters = {k: 0 for k in files.keys()}

    parquet_file = pq.ParquetFile(parquet_path)

    for batch in parquet_file.iter_batches(batch_size=500000):
        # to_pylist() is fast
        records = batch.to_pylist()
        for row in records:
            split = row.get("split")
            if split not in ["train", "validation"]:
                continue

            split_key = "val" if split == "validation" else "train"
            r_type = row.get("type")
            r_id = row.get("ID")

            base_record = {
                "dataset": "bigearthnet_txt",
                "split": split,
                "sample_id": f"ben_{r_id}",
                "image_available": False,
                "requires_external_imagery": True,
                "image_source": "BigEarthNet raw imagery required in Colab/cloud",
                "s1_name": row.get("s1_name", ""),
                "patch_id": row.get("patch_id", ""),
                "original_type": r_type,
                "category": row.get("category", "")
            }

            if r_type in ["binary", "mcq"]:
                base_record["task"] = "vqa"
                base_record["question"] = row.get("input")
                base_record["answer"] = row.get("output")
                files[f"vqa_{split_key}"].write(json.dumps(base_record) + "\n")
                counters[f"vqa_{split_key}"] += 1

            elif r_type == "captioning":
                base_record["task"] = "captioning"
                base_record["instruction"] = row.get("input")
                base_record["caption"] = row.get("output")
                files[f"caption_{split_key}"].write(json.dumps(base_record) + "\n")
                counters[f"caption_{split_key}"] += 1

            elif r_type == "bounding box":
                bbox = parse_ben_bbox(row.get("output"))
                if bbox:
                    base_record["task"] = "grounding"
                    base_record["text"] = row.get("input")
                    base_record["bbox"] = bbox
                    files[f"grounding_{split_key}"].write(json.dumps(base_record) + "\n")
                    counters[f"grounding_{split_key}"] += 1

    for f in files.values():
        f.close()

    for k, v in counters.items():
        print(f"BigEarthNet Prep: Extracted {v} {k}")

if __name__ == "__main__":
    process_bigearthnet()
