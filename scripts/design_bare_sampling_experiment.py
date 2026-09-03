import os
import sys
import math

sys.path.append(os.getcwd())
from training.scripts.train_optical_sar import SEN12MSSegmentationDataset
from scripts.optical_sar_sampling import get_train_sampling_weights

def main():
    print("Starting Bare Sampling Design Comparison...")

    base_dir = os.path.join(os.getcwd(), 'datasets/sen12ms')
    train_ds = SEN12MSSegmentationDataset(base_dir, split="train")
    
    print("Scanning training datasets to collect verified strata counts via get_patch_stratum()...")
    _, _, strata_counts, _ = get_train_sampling_weights(train_ds)
    
    total_train_patches = sum(strata_counts.values())
    epoch_samples = 800
    
    candidates = {
        "Candidate A - current baseline": {
            'vegetation_dominant': 0.45,
            'urban_rich': 0.25,
            'water_rich': 0.15,
            'mixed_minority': 0.10,
            'bare_rich': 0.05
        },
        "Candidate B - bare-focused": {
            'vegetation_dominant': 0.40,
            'urban_rich': 0.22,
            'water_rich': 0.13,
            'mixed_minority': 0.10,
            'bare_rich': 0.15
        }
    }

    for name, config in candidates.items():
        print(f"\n==========================================")
        print(f"--- {name} ---")
        print(f"==========================================")
        
        total_pct = sum(config.values())
        print(f"Total Probability Sum: {total_pct * 100.0:.2f}%")
        assert math.isclose(total_pct, 1.0, rel_tol=1e-5), f"{name} does not sum to 100%!"
        
        for stratum, target_prob in config.items():
            count = strata_counts[stratum]
            natural_prob = count / total_train_patches
            
            weight = target_prob / count
            oversample_factor = target_prob / natural_prob
            expected_draws = target_prob * epoch_samples
            
            print(f"\nStratum: {stratum}")
            print(f"  Target Percentage:       {target_prob * 100.0:.1f}%")
            print(f"  Source Patches:          {count}")
            print(f"  Sampling Weight:         {weight:.8f}")
            print(f"  Oversampling Factor:     {oversample_factor:.2f}x")
            print(f"  Expected Patches/800:    {expected_draws:.1f}")

    print("\nSTEP 4AB BARE SAMPLING DESIGN PASS")

if __name__ == '__main__':
    main()
