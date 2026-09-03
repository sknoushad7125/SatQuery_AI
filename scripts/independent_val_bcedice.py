import json
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from src.datasets.levir import LEVIRCDDataset
from src.models.baseline import SiameseChangeDetector
from src.metrics import ChangeDetectionMetrics

def main():
    print("--- INDEPENDENT VALIDATION (BCE + DICE MODEL) ---")
    
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Device: {device}")
    
    model = SiameseChangeDetector(encoder_name="resnet18", pretrained=False).to(device)
    model.load_state_dict(torch.load("checkpoints/best_bce_dice.pth", map_location=device))
    model.eval()
    
    dataset = LEVIRCDDataset("datasets/processed/manifests/levir_cd_manifest.json", split="val", crop_size=1024, is_train=False)
    loader = DataLoader(dataset, batch_size=1, shuffle=False)
    
    print(f"Validation pairs loaded: {len(dataset)}")
    
    metrics = ChangeDetectionMetrics()
    total_preds_pos = 0
    total_gt_pos = 0
    total_pixels = 0
    
    with torch.no_grad():
        for batch in tqdm(loader, desc="Validating"):
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
    
    # Check if predictions are all zero
    pred_pos_pct = (total_preds_pos / total_pixels) * 100
    gt_pos_pct = (total_gt_pos / total_pixels) * 100
    
    res["predicted_positive_pct"] = pred_pos_pct
    res["ground_truth_positive_pct"] = gt_pos_pct
    
    print(f"\nIoU: {res['iou']:.4f}")
    print(f"F1: {res['f1']:.4f}")
    print(f"Precision: {res['precision']:.4f}")
    print(f"Recall: {res['recall']:.4f}")
    print(f"Predicted Positive %: {pred_pos_pct:.4f}%")
    print(f"Ground Truth Positive %: {gt_pos_pct:.4f}%")
    
    import os
    os.makedirs("results/change_detection", exist_ok=True)
    with open("results/change_detection/independent_validation.json", "w") as f:
        json.dump(res, f, indent=4)
        
if __name__ == "__main__":
    main()
