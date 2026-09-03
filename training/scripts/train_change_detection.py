import argparse
import os
import time
import sys
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
import json
import torchvision.transforms as T
from PIL import Image

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
from backend.tools.change.real_change import SiamUNet_conc

class LEVIRCDDataset(Dataset):
    def __init__(self, data_dir, split="train", transform=None):
        self.data_dir = data_dir
        self.split = split
        self.transform = transform
        
        self.A_dir = os.path.join(data_dir, split, "A")
        self.B_dir = os.path.join(data_dir, split, "B")
        self.label_dir = os.path.join(data_dir, split, "label")
        
        if not os.path.exists(self.A_dir) or not os.path.exists(self.B_dir):
            raise FileNotFoundError(f"LEVIR-CD dataset structure not found in {data_dir}")
            
        self.samples = sorted(os.listdir(self.A_dir))
        
    def __len__(self):
        return len(self.samples)
        
    def __getitem__(self, idx):
        filename = self.samples[idx]
        img_A = Image.open(os.path.join(self.A_dir, filename)).convert('RGB')
        img_B = Image.open(os.path.join(self.B_dir, filename)).convert('RGB')
        label = Image.open(os.path.join(self.label_dir, filename)).convert('L')
        
        if self.transform:
            img_A = self.transform(img_A)
            img_B = self.transform(img_B)
            # Resize and convert label to tensor of 0s and 1s
            label = T.Resize((256, 256))(label)
            label = torch.from_numpy(np.array(label)).long()
            label = (label > 128).long()
            
        return img_A, img_B, label

class MockLEVIRCDDataset(Dataset):
    def __init__(self, num_samples=100, transform=None):
        self.num_samples = num_samples
    def __len__(self): return self.num_samples
    def __getitem__(self, idx):
        return torch.rand(3, 256, 256), torch.rand(3, 256, 256), torch.randint(0, 2, (256, 256), dtype=torch.long)

def compute_metrics(preds, labels):
    intersection = torch.logical_and(preds == 1, labels == 1).sum().item()
    union = torch.logical_or(preds == 1, labels == 1).sum().item()
    iou = intersection / union if union > 0 else 1.0
    
    tp = intersection
    fp = torch.logical_and(preds == 1, labels == 0).sum().item()
    fn = torch.logical_and(preds == 0, labels == 1).sum().item()
    
    precision = tp / (tp + fp) if (tp + fp) > 0 else 1.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 1.0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 1.0
    
    return iou, precision, recall, f1

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_path", type=str, default="/app/datasets/levir_cd")
    parser.add_argument("--output_dir", type=str, default="/app/training/checkpoints/siamunet_cd")
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--learning_rate", type=float, default=2e-4)
    parser.add_argument("--synthetic-dev-mode", action="store_true")
    args = parser.parse_args()

    if not args.synthetic_dev_mode and not os.path.exists(args.dataset_path):
        print("Dataset not found.\n")
        print("Training cannot proceed because a real remote-sensing dataset")
        print("is required for SIH26167 validation.\n")
        print("See docs/DATASET_SETUP.md")
        sys.exit(1)
        
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = SiamUNet_conc().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)
    
    transform = T.Compose([
        T.Resize((256, 256)),
        T.ToTensor()
    ])
    
    if args.synthetic_dev_mode:
        print("WARNING: Using DEV_TEST_ONLY synthetic dataset.")
        train_dataset = MockLEVIRCDDataset(num_samples=64, transform=transform)
        val_dataset = MockLEVIRCDDataset(num_samples=16, transform=transform)
        dataset_name = "Synthetic_DEV_TEST_ONLY"
    else:
        print(f"Loading real LEVIR-CD dataset from {args.dataset_path}")
        train_dataset = LEVIRCDDataset(args.dataset_path, split="train", transform=transform)
        val_dataset = LEVIRCDDataset(args.dataset_path, split="val", transform=transform)
        dataset_name = "LEVIR-CD"
        
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)
    
    best_iou = 0.0
    best_f1 = 0.0
    for epoch in range(args.epochs):
        model.train()
        train_loss = 0.0
        for img1, img2, labels in train_loader:
            img1, img2, labels = img1.to(device), img2.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(img1, img2)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
            
        model.eval()
        val_iou, val_f1, val_prec, val_rec = 0.0, 0.0, 0.0, 0.0
        with torch.no_grad():
            for img1, img2, labels in val_loader:
                img1, img2, labels = img1.to(device), img2.to(device), labels.to(device)
                outputs = model(img1, img2)
                preds = torch.argmax(outputs, dim=1)
                for p, l in zip(preds, labels):
                    iou, prec, rec, f1 = compute_metrics(p, l)
                    val_iou += iou; val_prec += prec; val_rec += rec; val_f1 += f1
                    
        num_val = len(val_dataset)
        avg_iou = val_iou / num_val
        avg_f1 = val_f1 / num_val
        
        print(f"Epoch {epoch+1} | Loss: {train_loss/len(train_loader):.4f} | Val IoU: {avg_iou:.4f} | Val F1: {avg_f1:.4f}")
        if avg_iou > best_iou:
            best_iou = avg_iou
            best_f1 = avg_f1
            
    os.makedirs(args.output_dir, exist_ok=True)
    ckpt_path = os.path.join(args.output_dir, "siamunet_cd.pth")
    torch.save(model.state_dict(), ckpt_path)
    
    metadata = {
        "exact_dataset": dataset_name,
        "number_of_training_samples": len(train_dataset),
        "number_of_validation_samples": len(val_dataset),
        "validation_IoU": best_iou,
        "validation_F1": best_f1,
        "date": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "model_version": "1.0.0"
    }
    with open(os.path.join(args.output_dir, "metadata.json"), "w") as f:
        json.dump(metadata, f, indent=2)

if __name__ == "__main__":
    main()
