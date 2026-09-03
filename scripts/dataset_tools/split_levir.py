#!/usr/bin/env python3
import os
import shutil
import random
from pathlib import Path

random.seed(42)

def enforce_splits(base_dir):
    """
    Splits LEVIR-CD or similar into strict 70/15/15 train/val/test splits,
    preventing any leakage across original pairs.
    """
    base = Path(base_dir)
    if not base.exists():
        print(f"Directory {base_dir} not found. Awaiting authentic download.")
        return

    # Assume we downloaded everything into datasets/levir_cd/raw/ (A, B, label)
    raw_a = base / "raw" / "A"
    raw_b = base / "raw" / "B"
    raw_label = base / "raw" / "label"
    
    if not raw_a.exists():
        print(f"Raw directory structure not found at {raw_a}.")
        return

    files = sorted([f.name for f in raw_a.glob("*.png")])
    if not files:
        print("No files found.")
        return

    print(f"Found {len(files)} authentic image pairs.")
    
    # Shuffle for splitting
    random.shuffle(files)
    
    n_total = len(files)
    n_train = int(n_total * 0.70)
    n_val = int(n_total * 0.15)
    
    splits = {
        "train": files[:n_train],
        "val": files[n_train:n_train+n_val],
        "test": files[n_train+n_val:]
    }

    print(f"Splitting strategy -> Train: {len(splits['train'])}, Val: {len(splits['val'])}, Test: {len(splits['test'])}")

    for split_name, split_files in splits.items():
        split_dir = base / split_name
        for sub in ["A", "B", "label"]:
            (split_dir / sub).mkdir(parents=True, exist_ok=True)
            
        for f_name in split_files:
            # Copy authentic files, preserving all bits
            shutil.copy2(raw_a / f_name, split_dir / "A" / f_name)
            shutil.copy2(raw_b / f_name, split_dir / "B" / f_name)
            shutil.copy2(raw_label / f_name, split_dir / "label" / f_name)

    print("Authentic split creation complete.")

if __name__ == "__main__":
    enforce_splits("datasets/levir_cd")
