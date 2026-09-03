import os
import sys
import time
import torch
import numpy as np
from torch.utils.data import DataLoader, WeightedRandomSampler

sys.path.append(os.getcwd())
from backend.tools.optical_sar.real_fusion import OpticalSARFusionNet
from training.scripts.train_optical_sar import SEN12MSSegmentationDataset, get_class_weights
from training.scripts.segmentation_loss import OpticalSARLoss
from scripts.optical_sar_sampling import get_train_sampling_weights
from training.scripts.segmentation_metrics import SegmentationMetrics

def main():
    device = torch.device("mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    base_dir = os.path.join(os.getcwd(), 'datasets/sen12ms')
    
    train_ds = SEN12MSSegmentationDataset(base_dir, split="train")
    val_ds = SEN12MSSegmentationDataset(base_dir, split="val")
    
    weights, _, targets, _ = get_train_sampling_weights(train_ds)
    print("Using Stratified Sampler Targets:")
    print(targets)
    
    sampler = WeightedRandomSampler(weights=weights, num_samples=len(train_ds), replacement=True)
    
    train_loader = DataLoader(train_ds, batch_size=8, sampler=sampler, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=8, shuffle=False, num_workers=0)
    
    model = OpticalSARFusionNet(num_classes=4).to(device)
    class_weights = get_class_weights().to(device)
    criterion = OpticalSARLoss(class_weights=class_weights)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    
    epochs = 3
    max_train_batches = 100
    checkpoint_path = "training/checkpoints/optical_sar_fullval_baseline.pth"
    os.makedirs(os.path.dirname(checkpoint_path), exist_ok=True)
    
    class_names = ['vegetation', 'built-up area', 'water body', 'bare land']
    
    best_miou = -1.0
    
    for epoch in range(epochs):
        print(f"\n==========================================")
        print(f"--- Epoch {epoch+1}/{epochs} ---")
        
        # TRAIN
        model.train()
        train_loss = 0.0
        
        start_time = time.time()
        for batch_idx, (s2, s1, mask) in enumerate(train_loader):
            if batch_idx >= max_train_batches:
                break
                
            s2, s1, mask = s2.to(device), s1.to(device), mask.to(device)
            
            optimizer.zero_grad()
            logits = model(s2, s1)
            loss = criterion(logits, mask)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            
        train_loss /= max_train_batches
        
        # VAL
        model.eval()
        val_loss = 0.0
        metrics = SegmentationMetrics(num_classes=4, class_names=class_names)
        val_patches = 0
        
        print(f"Evaluating Full Validation Set...")
        with torch.no_grad():
            for batch_idx, (s2, s1, mask) in enumerate(val_loader):
                s2, s1, mask = s2.to(device), s1.to(device), mask.to(device)
                
                logits = model(s2, s1)
                loss = criterion(logits, mask)
                val_loss += loss.item()
                
                metrics.update(logits, mask)
                val_patches += mask.size(0)
                
                if (batch_idx + 1) % 200 == 0:
                    print(f"  Validated {val_patches} patches...")
                
        val_loss /= len(val_loader)
        results = metrics.compute()
        
        print(f"Patches Evaluated: {val_patches} / 6130")
        print(f"Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")
        print(f"Pixel Accuracy: {results['pixel_accuracy']:.4f}")
        print(f"mIoU: {results['mIoU']:.4f}")
        print(f"mDice: {results['mDice']:.4f}")
        
        for c_name in class_names:
            iou = results['per_class_iou'][c_name]
            dice = results['per_class_dice'][c_name]
            print(f"  {c_name.title()}: IoU={iou:.4f} | Dice={dice:.4f}")
            
        if results['mIoU'] > best_miou:
            best_miou = results['mIoU']
            checkpoint = {
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'epoch': epoch,
                'val_loss': val_loss,
                'mIoU': results['mIoU'],
                'classes': class_names,
                'model_name': 'DualUNet-FeatureFusion'
            }
            torch.save(checkpoint, checkpoint_path)
            print(f"*** Saved New Best Checkpoint: {checkpoint_path} ***")
            
        print(f"Epoch Time: {time.time() - start_time:.2f}s")
        
    print("\nSTEP 4AJ FULL-VALIDATION BASELINE PASS")

if __name__ == '__main__':
    main()
