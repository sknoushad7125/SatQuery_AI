import sys
import os
import time
import torch
from torch.utils.data import DataLoader

sys.path.append(os.getcwd())
from training.scripts.train_optical_sar import SEN12MSSegmentationDataset, get_class_weights
from backend.tools.optical_sar.real_fusion import OpticalSARFusionNet
from training.scripts.segmentation_loss import OpticalSARLoss
from training.scripts.segmentation_metrics import SegmentationMetrics

def main():
    print("1. Initializing evaluation script...")
    device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Device: {device}")

    # Load dataset
    base_dir = os.path.join(os.getcwd(), 'datasets/sen12ms')
    val_ds = SEN12MSSegmentationDataset(base_dir, split="val")
    val_loader = DataLoader(val_ds, batch_size=2, shuffle=False, num_workers=0)
    
    total_patches = len(val_ds)
    print(f"Validation patches to evaluate: {total_patches}")
    
    # Load model
    ckpt_path = "training/checkpoints/optical_sar_smoke.pth"
    print(f"Loading checkpoint: {ckpt_path}")
    model = OpticalSARFusionNet(num_classes=4).to(device)
    ckpt = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(ckpt['model_state_dict'])
    model.eval()

    # Load metrics & loss
    classes = ckpt.get('classes', ["vegetation", "built-up area", "water body", "bare land"])
    weights = get_class_weights().to(device)
    criterion = OpticalSARLoss(class_weights=weights)
    metrics = SegmentationMetrics(num_classes=4, class_names=classes)

    print("2. Starting evaluation loop...")
    start_time = time.time()
    val_loss = 0.0
    
    with torch.no_grad():
        for batch_idx, (s2, s1, mask) in enumerate(val_loader):
            s2, s1, mask = s2.to(device), s1.to(device), mask.to(device)
            
            logits = model(s2, s1)
            loss = criterion(logits, mask)
            
            val_loss += loss.item()
            metrics.update(logits, mask)
            
            if (batch_idx + 1) % 100 == 0 or (batch_idx + 1) == len(val_loader):
                print(f"  Processed {min((batch_idx + 1) * 2, total_patches)} / {total_patches} patches...")

    val_loss /= len(val_loader)
    results = metrics.compute()
    elapsed = time.time() - start_time
    
    print("\n--- FULL VALIDATION BASELINE RESULTS ---")
    print(f"Total Patches Evaluated: {total_patches} (matches expected 6130: {total_patches == 6130})")
    print(f"Time Taken: {elapsed:.2f}s")
    print(f"Validation Loss: {val_loss:.4f}")
    print(f"mIoU: {results['mIoU']:.4f}")
    print(f"mDice: {results['mDice']:.4f}")
    print(f"Pixel Accuracy: {results['pixel_accuracy']:.4f}")
    
    print("\nTarget vs Predicted Pixel Counts:")
    for i, cls in enumerate(classes):
        tgt_count = int(metrics.targets_counts[i].item())
        prd_count = int(metrics.preds_counts[i].item())
        print(f"  {cls}: Target={tgt_count:,} | Predicted={prd_count:,}")
        
    print("\nPer-class Metrics:")
    for cls in classes:
        print(f"  {cls}: IoU={results['per_class_iou'][cls]:.4f}, Dice={results['per_class_dice'][cls]:.4f}")

    print("\nFULL VALIDATION BASELINE PASS")

if __name__ == '__main__':
    main()
