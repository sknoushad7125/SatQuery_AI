import torch
import torch.nn as nn
from src.datasets.levir import LEVIRCDDataset
from src.models.baseline import SiameseChangeDetector
from src.metrics import ChangeDetectionMetrics
from src.losses import BCEDiceLoss

def main():
    print("--- 4B. TINY OVERFITTING SANITY TEST (BCE + DICE) ---")
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    model = SiameseChangeDetector(encoder_name="resnet18", pretrained=True).to(device)
    
    # Load 2 pairs
    dataset = LEVIRCDDataset("datasets/processed/manifests/levir_cd_manifest.json", split="train", crop_size=256, is_train=True)
    subset = [dataset[0], dataset[1]]
    
    from torch.utils.data import DataLoader
    loader = DataLoader(subset, batch_size=2, shuffle=False)
    
    criterion = BCEDiceLoss()
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
                
    # Evaluate
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
    main()
