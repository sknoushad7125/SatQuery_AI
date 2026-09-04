import json
import random
from pathlib import Path
from collections import defaultdict

def sample_records(records, max_count, seed=42):
    random.seed(seed)
    if len(records) > max_count:
        return random.sample(records, max_count)
    return records

def write_jsonl(records, out_path):
    with open(out_path, 'w') as f:
        for r in records:
            f.write(json.dumps(r) + "\n")

def process_vqa(in_dir, out_dir):
    train_recs = []
    val_recs = []

    # Read RSVQA
    with open(in_dir / 'rsvqa_train.jsonl') as f:
        train_recs.extend([json.loads(line) for line in f])
    with open(in_dir / 'rsvqa_internal_val.jsonl') as f:
        val_recs.extend([json.loads(line) for line in f])

    # Read VRSBench VQA
    with open(in_dir / 'vrsbench_vqa_train.jsonl') as f:
        train_recs.extend([json.loads(line) for line in f])
    with open(in_dir / 'vrsbench_vqa_val.jsonl') as f:
        val_recs.extend([json.loads(line) for line in f])

    # BigEarthNet is excluded because image_available is False locally and instructions say:
    # "Do not include BigEarthNet records whose image is unavailable locally unless the DataLoader is explicitly designed for external cloud imagery."
    # We will just stick to locally available RSVQA and VRSBench.

    train_subset = sample_records(train_recs, 200000)
    # Don't strictly need to sample val, but let's keep it reasonable or keep all
    val_subset = val_recs # Keep all val for robust eval

    write_jsonl(train_subset, out_dir / 'vqa_train.jsonl')
    write_jsonl(val_subset, out_dir / 'vqa_val.jsonl')
    print(f"VQA: {len(train_subset)} train, {len(val_subset)} val")

def process_captioning(in_dir, out_dir):
    train_recs = []
    val_recs = []

    with open(in_dir / 'vrsbench_caption_train.jsonl') as f:
        train_recs.extend([json.loads(line) for line in f])
    with open(in_dir / 'vrsbench_caption_val.jsonl') as f:
        val_recs.extend([json.loads(line) for line in f])

    train_subset = sample_records(train_recs, 75000)
    write_jsonl(train_subset, out_dir / 'caption_train.jsonl')
    write_jsonl(val_recs, out_dir / 'caption_val.jsonl')
    print(f"Captioning: {len(train_subset)} train, {len(val_recs)} val")

def process_grounding(in_dir, out_dir):
    train_recs = []
    val_recs = []

    # ONLY VRSBench
    with open(in_dir / 'vrsbench_grounding_train.jsonl') as f:
        for line in f:
            rec = json.loads(line)
            # Basic bbox validation as requested
            bbox = rec.get("bbox")
            if bbox and len(bbox) == 4 and all(0 <= v <= 1 for v in bbox):
                x1, y1, x2, y2 = bbox
                if x1 <= x2 and y1 <= y2:
                    train_recs.append(rec)

    with open(in_dir / 'vrsbench_grounding_val.jsonl') as f:
        for line in f:
            rec = json.loads(line)
            bbox = rec.get("bbox")
            if bbox and len(bbox) == 4 and all(0 <= v <= 1 for v in bbox):
                x1, y1, x2, y2 = bbox
                if x1 <= x2 and y1 <= y2:
                    val_recs.append(rec)

    train_subset = sample_records(train_recs, 150000)
    write_jsonl(train_subset, out_dir / 'grounding_train.jsonl')
    write_jsonl(val_recs, out_dir / 'grounding_val.jsonl')
    print(f"Grounding: {len(train_subset)} train, {len(val_recs)} val")

def process_cdvqa(in_dir, out_dir):
    # We only have cdvqa_train.jsonl which contains everything.
    # Group by image pair to prevent leakage
    recs = []
    with open(in_dir / 'cdvqa_train.jsonl') as f:
        recs = [json.loads(line) for line in f]

    by_pair = defaultdict(list)
    for r in recs:
        pair_key = f"{r['before_image']}_{r['after_image']}"
        by_pair[pair_key].append(r)

    pairs = list(by_pair.keys())
    random.seed(42)
    random.shuffle(pairs)

    split_idx = int(len(pairs) * 0.9)
    train_pairs = pairs[:split_idx]
    val_pairs = pairs[split_idx:]

    train_subset = []
    for p in train_pairs:
        train_subset.extend(by_pair[p])

    val_subset = []
    for p in val_pairs:
        for r in by_pair[p]:
            r['split'] = 'internal_val'
            val_subset.append(r)

    for r in train_subset:
        r['split'] = 'internal_train'

    train_subset = sample_records(train_subset, 30000)
    write_jsonl(train_subset, out_dir / 'change_vqa_train.jsonl')
    write_jsonl(val_subset, out_dir / 'change_vqa_val.jsonl')
    print(f"Change VQA: {len(train_subset)} train, {len(val_subset)} val")

def main():
    in_dir = Path("training_data/manifests")
    out_dir = Path("training_data/subsets")
    out_dir.mkdir(parents=True, exist_ok=True)

    process_vqa(in_dir, out_dir)
    process_captioning(in_dir, out_dir)
    process_grounding(in_dir, out_dir)
    process_cdvqa(in_dir, out_dir)

if __name__ == "__main__":
    main()
