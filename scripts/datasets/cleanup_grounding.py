import json
from pathlib import Path
from training_data.dataloaders import create_grounding_dataloader
import traceback

def is_valid(r):
    # Check text/query
    has_text = bool(r.get("text")) or bool(r.get("query"))
    if not has_text:
        return False

    # Check image
    if not r.get("image"):
        return False

    # Check bbox
    bbox = r.get("bbox")
    if not bbox or len(bbox) != 4:
        return False
    if not all(isinstance(v, (int, float)) and 0 <= v <= 1 for v in bbox):
        return False
    x1, y1, x2, y2 = bbox
    if x1 > x2 or y1 > y2:
        return False

    return True

def run():
    in_file = Path("training_data/subsets/grounding_train.jsonl")
    out_file = Path("training_data/subsets/grounding_train_clean.jsonl")

    orig_count = 0
    clean_count = 0
    removed_count = 0

    valid_records = []
    with open(in_file, 'r') as f:
        for line in f:
            orig_count += 1
            r = json.loads(line)
            if is_valid(r):
                valid_records.append(r)
                clean_count += 1
            else:
                removed_count += 1

    with open(out_file, 'w') as f:
        for r in valid_records:
            f.write(json.dumps(r) + "\n")

    print(f"original count: {orig_count}")
    print(f"removed count: {removed_count}")
    print(f"clean count: {clean_count}")

    # Run dataloader dry run against the cleaned file
    print("\n--- DataLoader Dry Run ---")
    try:
        dl = create_grounding_dataloader(str(out_file), batch_size=2)
        batch = next(iter(dl))
        print(f"records: {len(dl.dataset)}")
        print(f"batch images: {batch['images'].shape}")
        print(f"batch queries: {batch['queries']}")
        print(f"batch bboxes: {batch['bboxes'].shape}")
        print("DataLoader result: PASS")
    except Exception as e:
        print(f"DataLoader result: FAIL ({e})")
        traceback.print_exc()

if __name__ == "__main__":
    run()
