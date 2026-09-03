import os
import json
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.datasets.levir import LEVIRCDDataset
from src.models.baseline import SiameseChangeDetector
from src.metrics import ChangeDetectionMetrics

def set_seed(seed=42):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    if torch.backends.mps.is_available():
        torch.mps.manual_seed(seed)
    import numpy as np
    import random
    np.random.seed(seed)
    random.seed(seed)

def main():
    set_seed(42)
    
    config = {
        "batch_size": 4, # Small batch for rapid prototype
        "epochs": 5,     # Increased to 5 to allow Dice loss to shape minority class
        "lr": 1e-4,
        "crop_size": 256,
        "encoder": "resnet18"
    }
    
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Using device: {device}")
    
    train_dataset = LEVIRCDDataset("datasets/processed/manifests/levir_cd_manifest.json", split="train", crop_size=config["crop_size"], is_train=True)
    val_dataset = LEVIRCDDataset("datasets/processed/manifests/levir_cd_manifest.json", split="val", crop_size=config["crop_size"], is_train=False)
    
    train_loader = DataLoader(train_dataset, batch_size=config["batch_size"], shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False, num_workers=0)
    
    model = SiameseChangeDetector(encoder_name=config["encoder"], pretrained=True).to(device)
    
    from src.losses import BCEDiceLoss
    criterion = BCEDiceLoss()
    optimizer = optim.Adam(model.parameters(), lr=config["lr"])
    
    # We will just train standard without AMP for safety on MPS for now, as some ops might not be perfectly supported in half precision
    
    os.makedirs("checkpoints", exist_ok=True)
    
    metrics = ChangeDetectionMetrics()
    best_iou = 0.0
    history = []
    
    print("Starting Training...")
    for epoch in range(config["epochs"]):
        model.train()
        train_loss = 0
        
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{config['epochs']} [Train]")
        for batch in pbar:
            img_a = batch["image_a"].to(device)
            img_b = batch["image_b"].to(device)
            mask = batch["mask"].to(device)
            
            optimizer.zero_grad()
            out = model(img_a, img_b)
            
            loss = criterion(out, mask)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            pbar.set_postfix({"loss": loss.item()})
            
        train_loss /= len(train_loader)
        
        # Validation
        model.eval()
        metrics.reset()
        val_loss = 0
        
        with torch.no_grad():
            for batch in tqdm(val_loader, desc=f"Epoch {epoch+1}/{config['epochs']} [Val]"):
                img_a = batch["image_a"].to(device)
                img_b = batch["image_b"].to(device)
                mask = batch["mask"].to(device)
                
                out = model(img_a, img_b)
                loss = criterion(out, mask)
                val_loss += loss.item()
                
                preds = torch.sigmoid(out) > 0.5
                metrics.update(preds, mask)
                
        val_loss /= len(val_loader)
        val_metrics = metrics.compute()
        
        print(f"Epoch {epoch+1} - Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")
        print(f"Val IoU: {val_metrics['iou']:.4f}, F1: {val_metrics['f1']:.4f}")
        
        hist_entry = {
            "epoch": epoch+1,
            "train_loss": train_loss,
            "val_loss": val_loss,
            "val_metrics": val_metrics
        }
        history.append(hist_entry)
        
        if val_metrics["iou"] >= best_iou:
            best_iou = val_metrics["iou"]
            torch.save(model.state_dict(), "checkpoints/best_bce_dice.pth")
            print("Saved new best checkpoint.")
            
    os.makedirs("results/change_detection", exist_ok=True)
    with open("results/change_detection/bce_dice_metrics.json", "w") as f:
        json.dump({"config": config, "history": history}, f, indent=2)

if __name__ == "__main__":
    main()
