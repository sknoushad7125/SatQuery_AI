import json
import torch
import torch.nn as nn
import numpy as np
from PIL import Image
from tqdm import tqdm
from src.datasets.levir import LEVIRCDDataset
from src.models.baseline import SiameseChangeDetector
from src.metrics import ChangeDetectionMetrics
import time

def measure_imbalance():
    print("--- 1. MEASURING CLASS IMBALANCE ---")
    with open("datasets/processed/manifests/levir_cd_manifest.json") as f:
        manifest = json.load(f)
    
    splits = {"train": [], "val": [], "test": []}
    for rec in manifest["records"].values():
        splits[rec["split"]].append(rec)
        
    for split_name, records in splits.items():
        print(f"\n{split_name.upper()} Split: {len(records)} pairs")
        
        # Subsample for speed if large, else measure all
        subsample = records[:50]
        
        tot_pix = 0
        changed_pix = 0
        
        for rec in tqdm(subsample, desc=f"Scanning {split_name} (max 50 imgs)"):
            lbl = np.array(Image.open(rec["label"]))
            lbl = (lbl > 127).astype(np.float32)
            tot_pix += lbl.size
            changed_pix += lbl.sum()
            
        print(f"Total Pixels (subsample): {tot_pix}")
        print(f"Changed Pixels: {changed_pix}")
        print(f"Changed %: {changed_pix / tot_pix * 100:.4f}%")
        print(f"Unchanged %: {(tot_pix - changed_pix) / tot_pix * 100:.4f}%")

def inspect_model_and_checkpoint():
    print("\n--- 2. INSPECTING MODEL AND CHECKPOINT ---")
    model = SiameseChangeDetector(encoder_name="resnet18", pretrained=False)
    num_params = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model parameters: {num_params} (Trainable: {trainable})")
    
    ckpt_path = "checkpoints/best_baseline.pth"
    import os
    if os.path.exists(ckpt_path):
        state = torch.load(ckpt_path, map_location="cpu")
        print("Checkpoint successfully loaded!")
        
        # Determine if it's untrained (are weights uniform or varied?)
        # Just check a layer's std
        layer_std = state["final_conv.0.weight"].std().item()
        print(f"Final Conv weight std: {layer_std:.4f}")
    else:
        print("No checkpoint found.")

def evaluate_baseline():
    print("\n--- 3. BASELINE EVALUATION ---")
    import os
    if not os.path.exists("checkpoints/best_baseline.pth"):
        print("Skipping evaluation, no checkpoint.")
        return
        
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    model = SiameseChangeDetector(encoder_name="resnet18", pretrained=False).to(device)
    model.load_state_dict(torch.load("checkpoints/best_baseline.pth", map_location=device))
    model.eval()
    
    val_dataset = LEVIRCDDataset("datasets/processed/manifests/levir_cd_manifest.json", split="val", crop_size=1024, is_train=False)
    from torch.utils.data import DataLoader
    val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False)
    
    metrics = ChangeDetectionMetrics()
    
    total_preds_pos = 0
    total_gt_pos = 0
    total_pixels = 0
    
    print("Running 10 samples from validation set for distribution check...")
    with torch.no_grad():
        for i, batch in enumerate(tqdm(val_loader)):
            if i >= 10: break
            img_a = batch["image_a"].to(device)
            img_b = batch["image_b"].to(device)
            mask = batch["mask"].to(device)
            
            out = model(img_a, img_b)
            preds = torch.sigmoid(out) > 0.5
            
            metrics.update(preds, mask)
            
            total_preds_pos += preds.sum().item()
            total_gt_pos += mask.sum().item()
            total_pixels += mask.numel()
            
    res = metrics.compute()
    print(f"IoU: {res['iou']:.4f}")
    print(f"F1: {res['f1']:.4f}")
    print(f"Prediction Positive %: {total_preds_pos / total_pixels * 100:.4f}%")
    print(f"Ground-Truth Positive %: {total_gt_pos / total_pixels * 100:.4f}%")

def overfitting_test():
    print("\n--- 4. TINY OVERFITTING SANITY TEST ---")
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    model = SiameseChangeDetector(encoder_name="resnet18", pretrained=True).to(device)
    
    # We load the exact same 2 pairs for 50 epochs
    dataset = LEVIRCDDataset("datasets/processed/manifests/levir_cd_manifest.json", split="train", crop_size=256, is_train=True)
    subset = [dataset[0], dataset[1]]
    
    # Custom collate
    from torch.utils.data import DataLoader
    loader = DataLoader(subset, batch_size=2, shuffle=False)
    
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    
    initial_loss = None
    final_loss = None
    
    metrics = ChangeDetectionMetrics()
    
    print("Training 2 samples for 50 iterations...")
    for epoch in range(50):
        model.train()
        for batch in loader:
            img_a = batch["image_a"].to(device)
            img_b = batch["image_b"].to(device)
            mask = batch["mask"].to(device)
            
            optimizer.zero_grad()
            out = model(img_a, img_b)
            loss = criterion(out, mask)
            loss.backward()
            optimizer.step()
            
            if epoch == 0:
                initial_loss = loss.item()
            if epoch == 49:
                final_loss = loss.item()
                
    # Evaluate on the same 2 pairs
    model.eval()
    with torch.no_grad():
        for batch in loader:
            img_a = batch["image_a"].to(device)
            img_b = batch["image_b"].to(device)
            mask = batch["mask"].to(device)
            
            out = model(img_a, img_b)
            preds = torch.sigmoid(out) > 0.5
            metrics.update(preds, mask)
            
            pos_pct = preds.sum().item() / mask.numel() * 100
            
    res = metrics.compute()
    print(f"Initial Loss: {initial_loss:.4f}")
    print(f"Final Loss: {final_loss:.4f}")
    print(f"Overfit IoU: {res['iou']:.4f}")
    print(f"Prediction Positive %: {pos_pct:.4f}%")

if __name__ == "__main__":
    measure_imbalance()
    inspect_model_and_checkpoint()
    evaluate_baseline()
    overfitting_test()
