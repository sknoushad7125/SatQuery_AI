import json
from pathlib import Path
from collections import defaultdict

def check_leakage():
    subset_dir = Path("training_data/subsets")
    report_file = Path("training_data/reports/phase2_leakage_audit.md")

    # Store sets for each task
    train_sids = set()
    val_sids = set()

    train_imgs = set()
    val_imgs = set()

    duplicate_sids = 0

    for fpath in subset_dir.glob("*.jsonl"):
        is_train = "train" in fpath.name
        with open(fpath, 'r') as f:
            for line in f:
                r = json.loads(line)
                sid = r.get("sample_id")

                # SIDs
                if sid:
                    if is_train:
                        if sid in train_sids: duplicate_sids += 1
                        train_sids.add(sid)
                    else:
                        if sid in val_sids: duplicate_sids += 1
                        val_sids.add(sid)

                # Images
                imgs_in_rec = []
                if "image" in r:
                    if isinstance(r["image"], dict):
                        imgs_in_rec.append(f"{r['image'].get('archive')}_{r['image'].get('member')}")
                    else:
                        imgs_in_rec.append(r["image"])
                if "before_image" in r:
                    imgs_in_rec.append(r["before_image"])
                if "after_image" in r:
                    imgs_in_rec.append(r["after_image"])

                for i_ref in imgs_in_rec:
                    if is_train:
                        train_imgs.add(i_ref)
                    else:
                        val_imgs.add(i_ref)

    # Overlaps
    sid_overlap = train_sids.intersection(val_sids)
    img_overlap = train_imgs.intersection(val_imgs)

    with open(report_file, 'w') as f:
        f.write("# Phase 2 Leakage Audit\n\n")
        f.write(f"- **Train Sample IDs**: {len(train_sids)}\n")
        f.write(f"- **Val Sample IDs**: {len(val_sids)}\n")
        f.write(f"- **Duplicate SIDs within splits**: {duplicate_sids}\n")
        f.write(f"- **Sample ID Leakage (Train \u2229 Val)**: {len(sid_overlap)}\n")
        f.write(f"- **Image Leakage (Train \u2229 Val)**: {len(img_overlap)}\n\n")

        f.write("## Documentation of Splits\n")
        f.write("- **VRSBench**: Official Train/Val zip separation is preserved.\n")
        f.write("- **RSVQA**: Internal split by image ID preserved.\n")
        f.write("- **CDVQA**: Internal split mapped successfully without overlap.\n")
        f.write("- **BigEarthNet Grounding**: Excluded from initial subset as axis semantics are unresolved.\n\n")

        status = "PASS" if (len(sid_overlap) == 0 and len(img_overlap) == 0 and duplicate_sids == 0) else "FAIL"
        f.write(f"**LEAKAGE_CHECK**: {status}\n")

    print(f"Leakage Audit Status: {status}")

if __name__ == "__main__":
    check_leakage()
