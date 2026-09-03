import argparse
import os
import time
import sys
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import numpy as np
import json
from glob import glob

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
from sen12ms_dataLoader import SEN12MSDataset as DataLoaderSEN12MS
from sen12ms_dataLoader import Seasons, S1Bands, S2Bands, LCBands
from backend.tools.optical_sar.real_fusion import OpticalSARFusionNet
from training.scripts.segmentation_loss import OpticalSARLoss
from training.scripts.segmentation_metrics import SegmentationMetrics


class SEN12MSSegmentationDataset(Dataset):
    def __init__(self, base_dir, split="train", config_path="scripts/config/optical_sar_preprocessing.json"):
        self.base_dir = base_dir
        self.split = split
        
        with open(config_path, 'r') as f:
            self.config = json.load(f)
            
        self.vv_min = self.config['S1']['VV']['p1']
        self.vv_max = self.config['S1']['VV']['p99']
        self.vh_min = self.config['S1']['VH']['p1']
        self.vh_max = self.config['S1']['VH']['p99']
        
        with open("scripts/config/optical_sar_split.json", "r") as f:
            splits = json.load(f)
            
        if split == "train": self.scenes = set(splits["train"])
        elif split == "val": self.scenes = set(splits["val"])
        elif split == "test": self.scenes = set(splits["test"])
        else: raise ValueError(f"Unknown split: {split}")
        
        self.loader = DataLoaderSEN12MS(base_dir)
        
        lc_files = glob(os.path.join(base_dir, 'ROIs1158_spring', 'lc_*', '*.tif'))
        self.samples = []
        for f in lc_files:
            scene_id = int(os.path.basename(f).replace(".tif", "").split("lc_")[1].split("_p")[0])
            if scene_id in self.scenes:
                patch_id = int(os.path.basename(f).replace(".tif", "").split("_p")[1])
                self.samples.append((scene_id, patch_id))
        self.samples.sort()
        
        self.class_map = np.zeros(18, dtype=np.int64)
        for i in [1,2,3,4,5,6,7,8,9,10,11,12,14]: self.class_map[i] = 0
        self.class_map[13] = 1
        self.class_map[17] = 2
        for i in [15,16]: self.class_map[i] = 3
        
    def __len__(self):
        return len(self.samples)
        
    def __getitem__(self, idx):
        scene_id, patch_id = self.samples[idx]
        try:
            s1, s2, lc, _ = self.loader.get_s1s2lc_triplet(
                Seasons.SPRING, scene_id, patch_id,
                s1_bands=S1Bands.ALL, s2_bands=S2Bands.ALL, lc_bands=LCBands.IGBP
            )
        except Exception as e:
            raise RuntimeError(f"Failed to load scene {scene_id} patch {patch_id}: {e}")
            
        s1 = torch.tensor(s1.astype(np.float32), dtype=torch.float32)
        vv = torch.clamp(s1[0:1], self.vv_min, self.vv_max)
        vv = (vv - self.vv_min) / (self.vv_max - self.vv_min)
        vh = torch.clamp(s1[1:2], self.vh_min, self.vh_max)
        vh = (vh - self.vh_min) / (self.vh_max - self.vh_min)
        s1_processed = torch.cat([vv, vh], dim=0)
        
        s2 = torch.tensor(s2.astype(np.float32), dtype=torch.float32)
        s2_processed = torch.clamp(s2 / 10000.0, 0.0, 1.0)
        
        lc = np.squeeze(lc, axis=0)
        mask = self.class_map[lc]
        mask_tensor = torch.tensor(mask, dtype=torch.long)
        
        if s1_processed.shape != (2, 256, 256): raise ValueError("Bad S1 shape")
        if s2_processed.shape != (13, 256, 256): raise ValueError("Bad S2 shape")
        if mask_tensor.shape != (256, 256): raise ValueError("Bad mask shape")
        
        return s2_processed, s1_processed, mask_tensor

def get_class_weights():
    counts = torch.tensor([1566218418, 248386131, 59011224, 1303651], dtype=torch.float32)
    total = counts.sum()
    frequencies = counts / total
    inv_sqrt_freq = 1.0 / torch.sqrt(frequencies)
    weights = inv_sqrt_freq / inv_sqrt_freq[0]
    return weights

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--max-train-batches", type=int, default=10)
    parser.add_argument("--max-val-batches", type=int, default=3)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--use-stratified-sampler", action="store_true", help="Enable Candidate A stratified sampling")
    parser.add_argument("--checkpoint", type=str, default="training/checkpoints/optical_sar_smoke.pth")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Using device: {device}")

    base_dir = os.path.join(os.getcwd(), 'datasets/sen12ms')
    train_ds = SEN12MSSegmentationDataset(base_dir, split="train")
    val_ds = SEN12MSSegmentationDataset(base_dir, split="val")
    
    from scripts.optical_sar_sampling import get_train_sampling_weights, get_patch_stratum
    if args.use_stratified_sampler:
        print("Stratified sampling ENABLED. Calculating weights...")
        weights, patch_strata, stratum_counts, targets = get_train_sampling_weights(train_ds)
        sampler = torch.utils.data.WeightedRandomSampler(
            weights=weights,
            num_samples=len(train_ds),
            replacement=True
        )
        train_loader = DataLoader(train_ds, batch_size=args.batch_size, sampler=sampler, num_workers=args.num_workers)
        print(f"Number of training patches: {len(train_ds)}")
        print(f"Expected Stratum Percentages: {targets}")
        print(f"Number of samples per epoch: {len(train_ds)}")
    else:
        print("Stratified sampling DISABLED.")
        train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)

    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)

    model = OpticalSARFusionNet(num_classes=4).to(device)
    weights = get_class_weights().to(device)
    criterion = OpticalSARLoss(class_weights=weights)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)

    best_val_miou = 0.0
    os.makedirs(os.path.dirname(args.checkpoint), exist_ok=True)

    classes = ["vegetation", "built-up area", "water body", "bare land"]

    for epoch in range(args.epochs):
        start_time = time.time()
        model.train()
        train_loss = 0.0
        sampled_strata = {'bare_rich': 0, 'water_rich': 0, 'urban_rich': 0, 'mixed_minority': 0, 'vegetation_dominant': 0}
        
        for batch_idx, (s2, s1, mask) in enumerate(train_loader):
            if batch_idx >= args.max_train_batches:
                break
                
            s2, s1, mask = s2.to(device), s1.to(device), mask.to(device)
            
            optimizer.zero_grad()
            logits = model(s2, s1)
            
            assert logits.shape == (s2.shape[0], 4, 256, 256), f"Bad output shape {logits.shape}"
            
            loss = criterion(logits, mask)
            if not torch.isfinite(loss):
                print(f"Warning: Non-finite loss at batch {batch_idx}. Skipping.")
                continue
                
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
            print(f"  Train Batch {batch_idx+1}/{args.max_train_batches} Loss: {loss.item():.4f}")
            
            if epoch == 0 and args.use_stratified_sampler:
                for m in mask:
                    st = get_patch_stratum(m.cpu().numpy())
                    sampled_strata[st] += 1

        train_loss /= min(len(train_loader), args.max_train_batches)
        if epoch == 0 and args.use_stratified_sampler:
            total_samples = sum(sampled_strata.values())
            if total_samples > 0:
                print(f"\nObserved sampling percentages (Epoch 0 Audit):")
                for k, v in sampled_strata.items():
                    print(f"  {k}: {v / total_samples * 100:.2f}%")


        model.eval()
        val_loss = 0.0
        metrics = SegmentationMetrics(num_classes=4, class_names=classes)
        with torch.no_grad():
            for batch_idx, (s2, s1, mask) in enumerate(val_loader):
                if batch_idx >= args.max_val_batches:
                    break
                s2, s1, mask = s2.to(device), s1.to(device), mask.to(device)
                logits = model(s2, s1)
                loss = criterion(logits, mask)
                val_loss += loss.item()
                metrics.update(logits, mask)
                print(f"  Val Batch {batch_idx+1}/{args.max_val_batches} Loss: {loss.item():.4f}")
        
        val_loss /= min(len(val_loader), args.max_val_batches)
        results = metrics.compute()
        
        elapsed = time.time() - start_time
        print(f"\n--- Epoch {epoch+1}/{args.epochs} Results ---")
        print(f"Time: {elapsed:.2f}s")
        print(f"Train Loss: {train_loss:.4f}")
        print(f"Val Loss:   {val_loss:.4f}")
        print(f"mIoU:       {results['mIoU']:.4f}")
        print(f"mDice:      {results['mDice']:.4f}")
        print(f"Pixel Acc:  {results['pixel_accuracy']:.4f}")
        for cls in classes:
            print(f"  {cls}: IoU={results['per_class_iou'][cls]:.4f}, Dice={results['per_class_dice'][cls]:.4f}")
        print("-----------------------------\n")
        
        if results['mIoU'] > best_val_miou:
            best_val_miou = results['mIoU']
            checkpoint = {
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'epoch': epoch,
                'val_loss': val_loss,
                'mIoU': results['mIoU'],
                'classes': classes,
                'model_name': 'DualUNet-FeatureFusion'
            }
            torch.save(checkpoint, args.checkpoint)
            print(f"Saved best checkpoint to {args.checkpoint}")

if __name__ == '__main__':
    main()
