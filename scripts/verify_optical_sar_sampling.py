import sys
import os
import math

sys.path.append(os.getcwd())
from training.scripts.train_optical_sar import SEN12MSSegmentationDataset
from scripts.optical_sar_sampling import get_train_sampling_weights

def verify():
    base_dir = os.path.join(os.getcwd(), 'datasets/sen12ms')
    print("1. Loading datasets...")
    train_ds = SEN12MSSegmentationDataset(base_dir, split="train")
    val_ds = SEN12MSSegmentationDataset(base_dir, split="val")
    
    num_train = len(train_ds)
    num_val = len(val_ds)
    
    assert num_train == 28609, f"Expected 28609 train patches, got {num_train}"
    assert num_val == 6130, f"Expected 6130 val patches, got {num_val}"
    
    print("2. Computing strata and weights for training set...")
    # Strict isolation: explicitly only pass train_ds
    weights, patch_strata, stratum_counts, targets = get_train_sampling_weights(train_ds)
    
    print("\n3. Validating basic checks...")
    assert len(weights) == num_train, "Weight list length does not match train patches!"
    assert len(patch_strata) == num_train, "Strata list length does not match train patches!"
    assert sum(stratum_counts.values()) == num_train, "Strata counts do not sum to total train patches!"
    assert all(math.isfinite(w) and w > 0 for w in weights), "Invalid weight values found!"
    
    print("\n4. Validating math distribution...")
    total_weight = sum(weights)
    
    for stratum, target_prob in targets.items():
        count = stratum_counts[stratum]
        stratum_weights = [w for w, s in zip(weights, patch_strata) if s == stratum]
        
        assert len(stratum_weights) == count, f"Mismatch in stratum weights count for {stratum}!"
        
        min_w = min(stratum_weights)
        max_w = max(stratum_weights)
        assert math.isclose(min_w, max_w, rel_tol=1e-5), f"Weights not uniform in stratum {stratum}"
        
        strat_total_weight = sum(stratum_weights)
        norm_prob = strat_total_weight / total_weight
        natural_freq = count / num_train
        oversample_factor = norm_prob / natural_freq
        
        print(f"\nStratum: {stratum}")
        print(f"  Patches: {count}")
        print(f"  Min Weight: {min_w:.8f}")
        print(f"  Max Weight: {max_w:.8f}")
        print(f"  Total Weight: {strat_total_weight:.4f}")
        print(f"  Normalized Prob: {norm_prob:.4f} (Expected: {target_prob:.4f})")
        print(f"  Oversampling Factor: {oversample_factor:.2f}x")
        
        assert math.isclose(norm_prob, target_prob, abs_tol=1e-4), f"Probability mismatch in {stratum}"
        
    print("\nAll conditions mathematically verified.")
    print("\nSTEP 4X SAMPLING WEIGHTS PASS")

if __name__ == '__main__':
    verify()
