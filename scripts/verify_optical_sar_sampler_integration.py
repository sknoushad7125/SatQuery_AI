import sys
import os
import torch
import math

sys.path.append(os.getcwd())
from training.scripts.train_optical_sar import SEN12MSSegmentationDataset
from scripts.optical_sar_sampling import get_train_sampling_weights

def verify():
    # Set deterministic seed
    torch.manual_seed(42)
    
    print("1. Loading real training dataset...")
    base_dir = os.path.join(os.getcwd(), 'datasets/sen12ms')
    train_ds = SEN12MSSegmentationDataset(base_dir, split="train")
    
    # Do NOT load validation or S1/S2 tensors, we just need the metadata
    print("2. Enabling sampler & calculating weights...")
    weights, patch_strata, stratum_counts, targets = get_train_sampling_weights(train_ds)
    
    print("3. Validating parameters...")
    assert len(weights) == 28609, f"Weights length is {len(weights)}, expected 28609"
    
    sampler = torch.utils.data.WeightedRandomSampler(
        weights=weights,
        num_samples=len(train_ds),
        replacement=True
    )
    assert len(sampler) == 28609, f"Sampler length is {len(sampler)}, expected 28609"
    assert sampler.replacement == True, "Sampler replacement must be True"
    
    print("4. Drawing one complete simulated sampled epoch...")
    # List of all 28609 indices drawn by the sampler in one epoch
    drawn_indices = list(sampler)
    
    # Count observed strata
    observed_counts = {k: 0 for k in targets.keys()}
    for idx in drawn_indices:
        stratum = patch_strata[idx]
        observed_counts[stratum] += 1
        
    print("\n--- Sampler Integration Results ---")
    total_drawn = sum(observed_counts.values())
    for stratum, expected_pct in targets.items():
        obs_count = observed_counts[stratum]
        obs_pct = obs_count / total_drawn
        diff = abs(obs_pct - expected_pct)
        
        print(f"{stratum}:")
        print(f"  Expected: {expected_pct*100:.1f}%")
        print(f"  Observed: {obs_pct*100:.2f}%")
        
        # Verify it is reasonably close (e.g. within 1.5% absolute difference)
        assert diff < 0.015, f"Stratum {stratum} deviates too much from target! Expected {expected_pct}, got {obs_pct}"

    print("\nSampler implementation distributes indices accurately.")
    print("STEP 4Y SAMPLER INTEGRATION PASS")

if __name__ == '__main__':
    verify()
