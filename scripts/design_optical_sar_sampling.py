import math

def main():
    print("==================================================")
    print("--- Optical-SAR Sampling Design Analysis ---")
    print("==================================================")

    # From Step 4V
    strata_counts = {
        'vegetation_dominant': 19025,
        'urban_rich': 6875,
        'water_rich': 1336,
        'mixed_minority': 1078,
        'bare_rich': 295
    }
    total_patches = 28609
    
    natural_freqs = {k: v / total_patches for k, v in strata_counts.items()}
    
    candidates = {
        "Candidate A (Moderate)": {
            'vegetation_dominant': 0.45,
            'urban_rich': 0.25,
            'water_rich': 0.15,
            'mixed_minority': 0.10,
            'bare_rich': 0.05
        },
        "Candidate B (Balanced)": {
            'vegetation_dominant': 0.40,
            'urban_rich': 0.25,
            'water_rich': 0.15,
            'mixed_minority': 0.10,
            'bare_rich': 0.10
        },
        "Candidate C (Conservative)": {
            'vegetation_dominant': 0.50,
            'urban_rich': 0.25,
            'water_rich': 0.12,
            'mixed_minority': 0.10,
            'bare_rich': 0.03
        }
    }
    
    num_samples = 1000
    
    for name, composition in candidates.items():
        print(f"\n[{name}]")
        for stratum, target_pct in composition.items():
            count = strata_counts[stratum]
            nat_pct = natural_freqs[stratum]
            
            target_samples = target_pct * num_samples
            oversample_factor = target_pct / nat_pct
            repeats_per_1000 = target_samples / count
            gt_10x = oversample_factor > 10.0
            
            print(f"  {stratum}:")
            print(f"    Target %:              {target_pct * 100.0:.1f}%")
            print(f"    Natural %:             {nat_pct * 100.0:.2f}%")
            print(f"    Available Source:      {count} patches")
            print(f"    Samples per 1000:      {target_samples:.1f}")
            print(f"    Oversampling Factor:   {oversample_factor:.2f}x")
            print(f"    >10x Oversampled?:     {gt_10x}")
            print(f"    Draws per patch / 1k:  {repeats_per_1000:.3f}")

    print("\n==================================================")
    print("--- Recommendation ---")
    print("Recommendation: Candidate A (Moderate)")
    print("\nReasoning:")
    print("1. Bare Exposure: Candidate A oversamples Bare ~4.8x (5% total). This provides a massive boost over its natural 1.03% occurrence, ensuring sufficient gradient updates without pushing the >9x overfitting territory seen in Candidate B (10% total, 9.7x).")
    print("2. Water Exposure: Water is cleanly boosted to 15% (3.2x oversample), safely ensuring enough representation.")
    print("3. Vegetation Foundation: By retaining 45% vegetation, the model will not forget the global background landscape, mitigating the risk of false positives while heavily lowering its ~66.5% natural dominance.")
    print("4. Avoids >10x Duplication: None of the strata in Candidate A cross the dangerous >10x oversampling threshold, preserving robust generalization.")
    
    print("\nSTEP 4W SAMPLING DESIGN PASS")

if __name__ == '__main__':
    main()
