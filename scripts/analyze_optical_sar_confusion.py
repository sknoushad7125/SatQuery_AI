import os
import sys
import torch
import time

sys.path.append(os.getcwd())
from backend.tools.optical_sar.real_fusion import OpticalSARFusionNet
from training.scripts.train_optical_sar import SEN12MSSegmentationDataset
from torch.utils.data import DataLoader

def main():
    print("Starting Confusion Matrix Analysis...")
    
    device = torch.device("mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    num_classes = 4
    model = OpticalSARFusionNet(num_classes=num_classes).to(device)
    checkpoint_path = "training/checkpoints/optical_sar_stratified_exp1.pth"
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    print(f"Loaded checkpoint from {checkpoint_path}")
    
    base_dir = os.path.join(os.getcwd(), 'datasets/sen12ms')
    val_ds = SEN12MSSegmentationDataset(base_dir, split="val")
    val_loader = DataLoader(val_ds, batch_size=2, shuffle=False, num_workers=0)
    
    conf_matrix = torch.zeros((num_classes, num_classes), dtype=torch.long, device=device)
    
    print("Evaluating 6,130 validation patches...")
    start_time = time.time()
    
    patches_evaluated = 0
    with torch.no_grad():
        for batch_idx, (s2, s1, mask) in enumerate(val_loader):
            s2, s1, mask = s2.to(device), s1.to(device), mask.to(device)
            
            logits = model(s2, s1)
            preds = torch.argmax(logits, dim=1)
            
            # Confusion matrix
            mask_flat = mask.view(-1)
            preds_flat = preds.view(-1)
            
            # indices = target * num_classes + pred
            indices = mask_flat * num_classes + preds_flat
            batch_conf = torch.bincount(indices, minlength=num_classes**2)
            conf_matrix += batch_conf.view(num_classes, num_classes)
            
            patches_evaluated += mask.size(0)
            if patches_evaluated % 1000 == 0:
                print(f"  Processed {patches_evaluated} / 6130 patches...")
                
    print("Evaluation complete.")
    print(f"Time: {time.time() - start_time:.2f}s")
    
    # 1. Raw Confusion Matrix
    print("\n--- 1. Raw Confusion Matrix (Row: GT, Col: Pred) ---")
    conf_matrix_cpu = conf_matrix.cpu().numpy()
    class_names = ["Vegetation", "Built-up", "Water", "Bare Land"]
    
    # Print header
    header = "GT \\ Pred  "
    for name in class_names:
        header += f"{name:>15}"
    print(header)
    
    for i in range(num_classes):
        row_str = f"{class_names[i]:>12}"
        for j in range(num_classes):
            row_str += f"{conf_matrix_cpu[i, j]:>15}"
        print(row_str)
        
    # 2. Row-normalized Confusion Matrix
    print("\n--- 2. Row-Normalized Confusion Matrix (%) ---")
    row_sums = conf_matrix_cpu.sum(axis=1, keepdims=True)
    norm_matrix = (conf_matrix_cpu / np.maximum(row_sums, 1)) * 100.0
    
    print(header)
    for i in range(num_classes):
        row_str = f"{class_names[i]:>12}"
        for j in range(num_classes):
            row_str += f"{norm_matrix[i, j]:>14.2f}%"
        print(row_str)

    # 3. Misclassification Destinations
    print("\n--- 3. Misclassification Destinations ---")
    for i in range(num_classes):
        total_gt = row_sums[i][0]
        correct = conf_matrix_cpu[i, i]
        incorrect = total_gt - correct
        print(f"{class_names[i]}: Total {total_gt} | Misclassified {incorrect} ({(incorrect/total_gt)*100:.2f}%)")
        for j in range(num_classes):
            if i != j:
                count = conf_matrix_cpu[i, j]
                if total_gt > 0:
                    print(f"  -> {class_names[j]}: {count} ({(count/total_gt)*100:.2f}%)")
                
    # 4. Specific Transitions
    print("\n--- 4. Specific Transitions Analysis ---")
    print(f"bare -> vegetation: {conf_matrix_cpu[3, 0]} pixels")
    print(f"bare -> built-up:   {conf_matrix_cpu[3, 1]} pixels")
    print(f"bare -> water:      {conf_matrix_cpu[3, 2]} pixels")
    print(f"vegetation -> bare: {conf_matrix_cpu[0, 3]} pixels")
    print(f"built-up -> bare:   {conf_matrix_cpu[1, 3]} pixels")
    print(f"water -> bare:      {conf_matrix_cpu[2, 3]} pixels")
    
    # 5. Total Validation Pixels Check
    total_pixels = conf_matrix_cpu.sum()
    expected_pixels = 6130 * 256 * 256
    print(f"\n--- 5. Total Pixels Check ---")
    print(f"Total evaluated pixels: {total_pixels}")
    print(f"Expected pixels (6130*256*256): {expected_pixels}")
    assert total_pixels == expected_pixels, "Pixel count mismatch!"
    print("Match: YES")
    
    # 6. Evaluation Check
    print(f"\n--- 6. Patches Evaluated ---")
    print(f"Total patches evaluated: {patches_evaluated}")
    assert patches_evaluated == 6130, "Patch count mismatch!"
    print("Match: YES")
    
    print("\nSTEP 4AD CONFUSION ANALYSIS PASS")

if __name__ == '__main__':
    import numpy as np
    main()
