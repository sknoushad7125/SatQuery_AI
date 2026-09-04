import json
from pathlib import Path
from collections import defaultdict

def get_img_ref(img_field):
    if isinstance(img_field, dict):
        return f"{img_field.get('archive')}_{img_field.get('member')}"
    return img_field

def run_audit():
    subsets = list(Path("training_data/subsets").glob("*.jsonl"))

    # 2. Composition
    composition = defaultdict(lambda: defaultdict(int))

    # 3. Completeness
    missing_fields = defaultdict(lambda: defaultdict(int))

    # 4. Duplicates
    seen_sids = defaultdict(set)
    duplicate_sids = defaultdict(int)
    seen_imgs = defaultdict(set)
    duplicate_imgs = defaultdict(int)
    seen_pairs = defaultdict(set)
    duplicate_pairs = defaultdict(int)

    # 5. Leakage
    train_sids = defaultdict(set)
    val_sids = defaultdict(set)
    train_imgs = defaultdict(set)
    val_imgs = defaultdict(set)
    train_pairs = defaultdict(set)
    val_pairs = defaultdict(set)

    for p in subsets:
        task = p.stem.replace("_train", "").replace("_val", "")
        split = "train" if "train" in p.name else "val"

        with open(p) as f:
            for line in f:
                r = json.loads(line)
                ds = r.get("dataset", "unknown")
                composition[p.name][ds] += 1

                # Check missing
                if task == "vqa":
                    if not r.get("question"): missing_fields[p.name]["question"] += 1
                    if r.get("answer") is None: missing_fields[p.name]["answer"] += 1
                elif task == "caption":
                    if not r.get("caption"): missing_fields[p.name]["caption"] += 1
                elif task == "grounding":
                    if not r.get("text") and not r.get("query"): missing_fields[p.name]["text"] += 1
                    if not r.get("bbox"): missing_fields[p.name]["bbox"] += 1
                elif task == "change_vqa":
                    if not r.get("before_image"): missing_fields[p.name]["before_image"] += 1
                    if not r.get("after_image"): missing_fields[p.name]["after_image"] += 1
                    if not r.get("question"): missing_fields[p.name]["question"] += 1
                    if r.get("answer") is None: missing_fields[p.name]["answer"] += 1

                # Check duplicates & leakage
                sid = r.get("sample_id")
                if sid:
                    if sid in seen_sids[p.name]:
                        duplicate_sids[p.name] += 1
                    seen_sids[p.name].add(sid)

                    if split == "train":
                        train_sids[task].add(sid)
                    else:
                        val_sids[task].add(sid)

                # Images
                img_refs = []
                if "image" in r:
                    img_refs.append(get_img_ref(r["image"]))
                if "before_image" in r:
                    img_refs.append(get_img_ref(r["before_image"]))
                if "after_image" in r:
                    img_refs.append(get_img_ref(r["after_image"]))

                for img in img_refs:
                    if img in seen_imgs[p.name]:
                        duplicate_imgs[p.name] += 1
                    seen_imgs[p.name].add(img)

                    if split == "train":
                        train_imgs[task].add(img)
                    else:
                        val_imgs[task].add(img)

                # Pair
                if task == "change_vqa":
                    pair = f"{r.get('before_image')}_{r.get('after_image')}"
                    if pair in seen_pairs[p.name]:
                        duplicate_pairs[p.name] += 1
                    seen_pairs[p.name].add(pair)

                    if split == "train":
                        train_pairs[task].add(pair)
                    else:
                        val_pairs[task].add(pair)

    print("=== COMPOSITION ===")
    for p, ds_counts in sorted(composition.items()):
        for ds, count in ds_counts.items():
            print(f"{p:25s} | {ds:15s} | {count}")

    print("\n=== MISSING ===")
    for p, fields in sorted(missing_fields.items()):
        for f, count in fields.items():
            print(f"{p:25s} | {f:15s} | {count}")

    print("\n=== DUPLICATES ===")
    for p in sorted([s.name for s in subsets]):
        print(f"{p:25s} | SIDs: {duplicate_sids[p]} | IMGs: {duplicate_imgs[p]} | PAIRs: {duplicate_pairs[p]}")

    print("\n=== LEAKAGE ===")
    for task in train_sids.keys():
        sid_leak = len(train_sids[task].intersection(val_sids[task]))
        img_leak = len(train_imgs[task].intersection(val_imgs[task]))
        print(f"{task:15s} | SID Leak: {sid_leak} | IMG Leak: {img_leak}")
        if task == "change_vqa":
            pair_leak = len(train_pairs[task].intersection(val_pairs[task]))
            print(f"{task:15s} | PAIR Leak: {pair_leak}")

if __name__ == "__main__":
    run_audit()
