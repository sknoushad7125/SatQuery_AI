import sys
import os
import torch
import numpy as np

sys.path.append(os.getcwd())
from training.scripts.train_optical_sar import SEN12MSSegmentationDataset

def verify():
    base_dir = os.path.join(os.getcwd(), 'datasets/sen12ms')
    
    print("1. Instantiating datasets...")
    train_ds = SEN12MSSegmentationDataset(base_dir, split="train")
    val_ds = SEN12MSSegmentationDataset(base_dir, split="val")
    test_ds = SEN12MSSegmentationDataset(base_dir, split="test")
    
    # Extract sets of scene IDs present
    train_scenes = set(s[0] for s in train_ds.samples)
    val_scenes = set(s[0] for s in val_ds.samples)
    test_scenes = set(s[0] for s in test_ds.samples)
    
    print(f"\n2. Dataset Sizes:")
    print(f"Train: {len(train_ds)} patches from {len(train_scenes)} scenes")
    print(f"Val:   {len(val_ds)} patches from {len(val_scenes)} scenes")
    print(f"Test:  {len(test_ds)} patches from {len(test_scenes)} scenes")
    
    print("\n3. Verifying scene isolation...")
    assert len(train_scenes.intersection(val_scenes)) == 0, "Leakage: Train overlaps Val"
    assert len(train_scenes.intersection(test_scenes)) == 0, "Leakage: Train overlaps Test"
    assert len(val_scenes.intersection(test_scenes)) == 0, "Leakage: Val overlaps Test"
    print("Zero scene overlap verified. Isolation is PASS.")
    
    def check_samples(ds, name):
        print(f"\n--- Checking 3 random samples from {name.upper()} split ---")
        np.random.seed(42)
        indices = np.random.choice(len(ds), 3, replace=False)
        for i in indices:
            s2, s1, mask = ds[i]
            
            # 4. Shapes and Dtypes
            assert s2.shape == (13, 256, 256), f"Bad S2 shape: {s2.shape}"
            assert s1.shape == (2, 256, 256), f"Bad S1 shape: {s1.shape}"
            assert mask.shape == (256, 256), f"Bad Mask shape: {mask.shape}"
            
            assert s2.dtype == torch.float32, f"Bad S2 dtype: {s2.dtype}"
            assert s1.dtype == torch.float32, f"Bad S1 dtype: {s1.dtype}"
            assert mask.dtype == torch.long, f"Bad Mask dtype: {mask.dtype}"
            
            # 5. Numerical ranges
            assert s2.min() >= 0.0 and s2.max() <= 1.0, f"S2 out of bounds: [{s2.min()}, {s2.max()}]"
            assert s1.min() >= 0.0 and s1.max() <= 1.0, f"S1 out of bounds: [{s1.min()}, {s1.max()}]"
            
            # 6. Mask IDs
            unique_ids = torch.unique(mask).tolist()
            assert all(uid in [0, 1, 2, 3] for uid in unique_ids), f"Invalid Mask IDs: {unique_ids}"
            
            # 8. Pixel counts
            counts = {c: int((mask == c).sum()) for c in range(4)}
            print(f"Sample index {i}: Shapes S2{list(s2.shape)} S1{list(s1.shape)} Mask{list(mask.shape)} | "
                  f"S1_range=[{s1.min():.2f}, {s1.max():.2f}] | S2_range=[{s2.min():.2f}, {s2.max():.2f}] | "
                  f"Mask Classes: {unique_ids} | Pixel Counts: Veg:{counts[0]} Urban:{counts[1]} Water:{counts[2]} Bare:{counts[3]}")

    check_samples(train_ds, "train")
    check_samples(val_ds, "val")
    check_samples(test_ds, "test")
    
    print("\nREAL DATASET PIPELINE PASS")

if __name__ == '__main__':
    verify()
