import os
import sys
import math
import torch
import time

sys.path.append(os.getcwd())
from backend.tools.optical_sar.real_fusion import OpticalSARFusionNet
from training.scripts.train_optical_sar import SEN12MSSegmentationDataset
from torch.utils.data import DataLoader

def main():
    print("Starting Diagnostic Pass...")
    
    device = torch.device("mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    model = OpticalSARFusionNet(num_classes=4).to(device)
    checkpoint_path = "training/checkpoints/optical_sar_stratified_bare_exp2.pth"
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    print(f"Loaded checkpoint from {checkpoint_path}")
    
    base_dir = os.path.join(os.getcwd(), 'datasets/sen12ms')
    val_ds = SEN12MSSegmentationDataset(base_dir, split="val")
    val_loader = DataLoader(val_ds, batch_size=2, shuffle=False, num_workers=0)
    
    num_classes = 4
    intersections = torch.zeros(num_classes, dtype=torch.long, device=device)
    unions = torch.zeros(num_classes, dtype=torch.long, device=device)
    target_counts = torch.zeros(num_classes, dtype=torch.long, device=device)
    pred_counts = torch.zeros(num_classes, dtype=torch.long, device=device)
    correct_pixels = 0
    total_pixels = 0
    
    class_patch_counts = {0: 0, 1: 0, 2: 0, 3: 0}
    
    print("Starting full validation evaluation (6,130 patches)...")
    start_time = time.time()
    
    with torch.no_grad():
        for batch_idx, (s2, s1, mask) in enumerate(val_loader):
            s2, s1, mask = s2.to(device), s1.to(device), mask.to(device)
            
            logits = model(s2, s1)
            preds = torch.argmax(logits, dim=1)
            
            correct_pixels += (preds == mask).sum().item()
            total_pixels += mask.numel()
            
            for c in range(num_classes):
                pred_c = (preds == c)
                target_c = (mask == c)
                
                target_counts[c] += target_c.sum()
                pred_counts[c] += pred_c.sum()
                intersections[c] += (pred_c & target_c).sum()
                unions[c] += (pred_c | target_c).sum()
                
                # Check per-patch presence in the batch
                for b in range(mask.size(0)):
                    if (mask[b] == c).any():
                        class_patch_counts[c] += 1
            
            if (batch_idx + 1) % 500 == 0:
                print(f"  Processed {min((batch_idx+1)*2, 6130)} / 6130 patches...")
                
    print("Evaluation complete.")
    print(f"Time: {time.time() - start_time:.2f}s")
    
    print("\n--- Diagnostic Results ---")
    ious = []
    dices = []
    class_names = ["Vegetation (0)", "Built-up (1)", "Water (2)", "Bare Land (3)"]
    
    for c in range(num_classes):
        intersection = intersections[c].item()
        union = unions[c].item()
        target = target_counts[c].item()
        pred = pred_counts[c].item()
        
        iou = intersection / union if union > 0 else float('nan')
        dice = (2.0 * intersection) / (pred + target) if (pred + target) > 0 else float('nan')
        
        ious.append(iou)
        dices.append(dice)
        
        print(f"\nClass: {class_names[c]}")
        print(f"  Target Pixels:    {target}")
        print(f"  Predicted Pixels: {pred}")
        print(f"  Intersection:     {intersection}")
        print(f"  Union:            {union}")
        if not math.isnan(iou):
            print(f"  IoU:              {iou:.4f}")
            print(f"  Dice:             {dice:.4f}")
        else:
            print("  IoU:              NaN")
            print("  Dice:             NaN")
        print(f"  Patches Present:  {class_patch_counts[c]}")
        print(f"  Any Pred Pixels?: {'Yes' if pred > 0 else 'No'}")
        
    pixel_acc = correct_pixels / total_pixels
    valid_ious = [x for x in ious if not math.isnan(x)]
    valid_dices = [x for x in dices if not math.isnan(x)]
    
    m_iou = sum(valid_ious) / len(valid_ious) if valid_ious else 0.0
    m_dice = sum(valid_dices) / len(valid_dices) if valid_dices else 0.0
    
    print(f"\nOverall Metrics:")
    print(f"  Pixel Accuracy: {pixel_acc:.4f}")
    print(f"  mIoU:           {m_iou:.4f}")
    print(f"  mDice:          {m_dice:.4f}")
    
    print("\nSTEP 4Z DIAGNOSTIC PASS")

if __name__ == '__main__':
    main()
