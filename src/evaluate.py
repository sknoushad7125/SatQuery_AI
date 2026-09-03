import os
import json
import time
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

from src.datasets.levir import LEVIRCDDataset
from src.models.baseline import SiameseChangeDetector
from src.metrics import ChangeDetectionMetrics

def visualize_prediction(img_a_path, img_b_path, gt_path, pred_mask, out_path):
    img_a = Image.open(img_a_path)
    img_b = Image.open(img_b_path)
    gt = Image.open(gt_path)
    
    fig, axs = plt.subplots(1, 4, figsize=(20, 5))
    axs[0].imshow(img_a)
    axs[0].set_title("Image A")
    axs[0].axis('off')
    
    axs[1].imshow(img_b)
    axs[1].set_title("Image B")
    axs[1].axis('off')
    
    axs[2].imshow(gt, cmap='gray')
    axs[2].set_title("Ground Truth")
    axs[2].axis('off')
    
    axs[3].imshow(pred_mask, cmap='gray')
    axs[3].set_title("Prediction")
    axs[3].axis('off')
    
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()

def main():
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Evaluating on device: {device}")
    
    model = SiameseChangeDetector(encoder_name="resnet18", pretrained=False).to(device)
    model.load_state_dict(torch.load("checkpoints/best_baseline.pth", map_location=device))
    model.eval()
    
    num_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {num_params:,}")
    
    metrics = ChangeDetectionMetrics()
    os.makedirs("results/visualizations", exist_ok=True)
    
    with open("datasets/processed/manifests/levir_cd_manifest.json") as f:
        manifest = json.load(f)
        
    test_records = [rec for rec in manifest["records"].values() if rec["split"] == "test"]
    
    test_dataset = LEVIRCDDataset("datasets/processed/manifests/levir_cd_manifest.json", split="test", crop_size=1024, is_train=False)
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)
    
    print("Running evaluation on test set...")
    start_time = time.time()
    with torch.no_grad():
        for i, batch in enumerate(tqdm(test_loader)):
            img_a = batch["image_a"].to(device)
            img_b = batch["image_b"].to(device)
            mask = batch["mask"].to(device)
            
            out = model(img_a, img_b)
            preds = torch.sigmoid(out) > 0.5
            
            metrics.update(preds, mask)
            
            if i < 5:
                pred_np = preds[0, 0].cpu().numpy().astype(np.uint8) * 255
                filename = batch["filename"][0]
                rec = [r for r in test_records if r["image_a"].endswith(filename)][0]
                
                visualize_prediction(
                    rec["image_a"], 
                    rec["image_b"], 
                    rec["label"], 
                    pred_np, 
                    os.path.join("results/visualizations", f"pred_{filename}")
                )
                
    end_time = time.time()
    inference_time = end_time - start_time
    fps = len(test_loader) / inference_time
    print(f"Inference Time: {inference_time:.2f}s ({fps:.2f} FPS)")
    
    test_metrics = metrics.compute()
    test_metrics["inference_fps"] = fps
    test_metrics["parameter_count"] = num_params
    
    print("Test Metrics:")
    for k, v in test_metrics.items():
        if isinstance(v, dict): continue
        print(f"{k}: {v:.4f}" if isinstance(v, float) else f"{k}: {v}")
        
    with open("results/test_metrics.json", "w") as f:
        json.dump(test_metrics, f, indent=2)
        
if __name__ == "__main__":
    main()
