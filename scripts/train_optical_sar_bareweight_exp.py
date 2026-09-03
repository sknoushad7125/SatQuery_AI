import os
import sys
import time
import torch
import numpy as np
import argparse
from torch.utils.data import DataLoader, WeightedRandomSampler

sys.path.append(os.getcwd())
from backend.tools.optical_sar.real_fusion import OpticalSARFusionNet
from training.scripts.train_optical_sar import SEN12MSSegmentationDataset
from training.scripts.segmentation_loss import OpticalSARLoss
from scripts.optical_sar_sampling import get_train_sampling_weights
from training.scripts.segmentation_metrics import SegmentationMetrics

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--weight", type=float, required=True)
    parser.add_argument("--checkpoint", type=str, required=True)
    args = parser.parse_args()

    bare_weight = args.weight
    checkpoint_path = args.checkpoint

    device = torch.device("mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n=======================================================")
    print(f"STARTING EXPERIMENT: Bare Weight = {bare_weight}")
    print(f"=======================================================")
    
    base_dir = os.path.join(os.getcwd(), 'datasets/sen12ms')
    
    train_ds = SEN12MSSegmentationDataset(base_dir, split="train")
    val_ds = SEN12MSSegmentationDataset(base_dir, split="val")
    
    weights, _, targets, _ = get_train_sampling_weights(train_ds)
    sampler = WeightedRandomSampler(weights=weights, num_samples=len(train_ds), replacement=True)
    
    train_loader = DataLoader(train_ds, batch_size=8, sampler=sampler, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=8, shuffle=False, num_workers=0)
    
    model = OpticalSARFusionNet(num_classes=4).to(device)
    class_weights = torch.tensor([1.0, 2.5111, 5.1518, bare_weight], dtype=torch.float32).to(device)
    criterion = OpticalSARLoss(class_weights=class_weights)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    
    epochs = 3
    max_train_batches = 100
    class_names = ['vegetation', 'built-up area', 'water body', 'bare land']
    
    best_miou = -1.0
    best_epoch = -1
    
    for epoch in range(epochs):
        print(f"\n--- Exp: BW={bare_weight} | Epoch {epoch+1}/{epochs} ---")
        
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
        
        with torch.no_grad():
            for batch_idx, (s2, s1, mask) in enumerate(val_loader):
                s2, s1, mask = s2.to(device), s1.to(device), mask.to(device)
                
                logits = model(s2, s1)
                loss = criterion(logits, mask)
                val_loss += loss.item()
                
                metrics.update(logits, mask)
                val_patches += mask.size(0)
                
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
            best_epoch = epoch + 1
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
        
    print(f"\nDONE Exp BW={bare_weight}. Best mIoU: {best_miou:.4f} at Epoch {best_epoch}.")

if __name__ == '__main__':
    main()
