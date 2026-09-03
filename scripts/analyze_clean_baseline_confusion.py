import os
import sys
import torch
import time
import numpy as np

sys.path.append(os.getcwd())
from backend.tools.optical_sar.real_fusion import OpticalSARFusionNet
from training.scripts.train_optical_sar import SEN12MSSegmentationDataset
from torch.utils.data import DataLoader

def main():
    print("Starting Clean Baseline Confusion Matrix Analysis...")
    
    device = torch.device("mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    num_classes = 4
    model = OpticalSARFusionNet(num_classes=num_classes).to(device)
    checkpoint_path = "training/checkpoints/optical_sar_fullval_baseline.pth"
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
            
            mask_flat = mask.view(-1)
            preds_flat = preds.view(-1)
            
            indices = mask_flat * num_classes + preds_flat
            batch_conf = torch.bincount(indices, minlength=num_classes**2)
            conf_matrix += batch_conf.view(num_classes, num_classes)
            
            patches_evaluated += mask.size(0)
            if patches_evaluated % 1000 == 0:
                print(f"  Processed {patches_evaluated} / 6130 patches...")
                
    print("Evaluation complete.")
    print(f"Time: {time.time() - start_time:.2f}s")
    
    conf_matrix_cpu = conf_matrix.cpu().numpy()
    class_names = ["Vegetation", "Built-up", "Water", "Bare Land"]
    
    print("\n--- 1. Raw Confusion Matrix (Row: GT, Col: Pred) ---")
    header = "GT \\ Pred  "
    for name in class_names:
        header += f"{name:>15}"
    print(header)
    for i in range(num_classes):
        row_str = f"{class_names[i]:>12}"
        for j in range(num_classes):
            row_str += f"{conf_matrix_cpu[i, j]:>15}"
        print(row_str)
        
    print("\n--- 2. Row-Normalized Confusion Matrix (%) ---")
    row_sums = conf_matrix_cpu.sum(axis=1, keepdims=True)
    col_sums = conf_matrix_cpu.sum(axis=0)
    norm_matrix = (conf_matrix_cpu / np.maximum(row_sums, 1)) * 100.0
    print(header)
    for i in range(num_classes):
        row_str = f"{class_names[i]:>12}"
        for j in range(num_classes):
            row_str += f"{norm_matrix[i, j]:>14.2f}%"
        print(row_str)
        
    print("\n--- 3. Class Metrics ---")
    ious = []
    for i in range(num_classes):
        target = row_sums[i][0]
        correct = conf_matrix_cpu[i, i]
        pred_count = col_sums[i]
        recall = (correct / target) * 100.0 if target > 0 else 0.0
        union = target + pred_count - correct
        iou = correct / union if union > 0 else 0.0
        ious.append(iou)
        print(f"{class_names[i]}: Target={target} | Correct={correct} | Recall={recall:.2f}% | Predicted={pred_count}")
        
    print("\n--- 4. Specific Transitions ---")
    print(f"bare -> vegetation: {conf_matrix_cpu[3, 0]} pixels")
    print(f"bare -> built-up:   {conf_matrix_cpu[3, 1]} pixels")
    print(f"bare -> water:      {conf_matrix_cpu[3, 2]} pixels")
    print(f"bare -> bare:       {conf_matrix_cpu[3, 3]} pixels")
    print(f"vegetation -> bare: {conf_matrix_cpu[0, 3]} pixels")
    print(f"built-up -> bare:   {conf_matrix_cpu[1, 3]} pixels")
    print(f"water -> bare:      {conf_matrix_cpu[2, 3]} pixels")
    
    print("\n--- 5. Column Totals (Predicted Frequencies) ---")
    for i in range(num_classes):
        print(f"{class_names[i]}: {col_sums[i]} pixels")
        
    print("\n--- 6. Matrix Total vs Expected ---")
    total_pixels = conf_matrix_cpu.sum()
    expected = 6130 * 256 * 256
    print(f"Total evaluated pixels: {total_pixels}")
    print(f"Expected pixels: {expected}")
    if total_pixels == expected:
        print("Match: YES")
    else:
        print("Match: NO")
        
    print("\n--- 7. Patches Evaluated ---")
    print(f"Evaluated patches: {patches_evaluated} / 6130")
    if patches_evaluated == 6130:
        print("Match: YES")
    else:
        print("Match: NO")
        
    print("\n--- 8. Derived IoU Comparison ---")
    print(f"Calculated mIoU: {np.mean(ious):.4f} (Expected: 0.6127)")
    print(f"Calculated Veg IoU: {ious[0]:.4f} (Expected: 0.9012)")
    print(f"Calculated Built-up IoU: {ious[1]:.4f} (Expected: 0.6682)")
    print(f"Calculated Water IoU: {ious[2]:.4f} (Expected: 0.8809)")
    print(f"Calculated Bare IoU: {ious[3]:.4f} (Expected: 0.0007)")
    
    print("\nSTEP 4AK CLEAN BASELINE CONFUSION PASS")

if __name__ == '__main__':
    main()
