import json
import os
import time
import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from torchvision import transforms
from src.models.vqa import SimpleRSVQAModel
from tqdm import tqdm

class RSVQA_TestDataset(Dataset):
    def __init__(self, split="val", max_samples=1000):
        with open(f'datasets/rsvqa/LR_split_{split}_questions.json') as f:
            self.q_data = json.load(f)["questions"]
        with open(f'datasets/rsvqa/LR_split_{split}_answers.json') as f:
            self.a_data = {a["id"]: a for a in json.load(f)["answers"]}
        with open(f'datasets/rsvqa/LR_split_{split}_images.json') as f:
            self.i_data = {i["id"]: i for i in json.load(f)["images"]}
            
        with open('datasets/processed/rsvqa/q_vocab.json') as f:
            self.q_vocab = json.load(f)
        with open('datasets/processed/rsvqa/a_vocab.json') as f:
            self.a_vocab = json.load(f)
            self.idx_to_ans = {v: k for k, v in self.a_vocab.items()}
            
        self.samples = []
        for q in self.q_data:
            if not q.get("active", False) or "question" not in q: continue
            if len(q["answers_ids"]) == 0: continue
            ans_id = q["answers_ids"][0]
            if ans_id not in self.a_data: continue
            
            ans_str = str(self.a_data[ans_id]["answer"])
            if ans_str not in self.a_vocab: continue
            
            img_id = q["img_id"]
            if img_id not in self.i_data: continue
            
            img_path = os.path.join('datasets/rsvqa/Images_LR', f"{img_id}.tif")
            if not os.path.exists(img_path): continue
            
            self.samples.append({
                "question": q["question"],
                "answer_id": self.a_vocab[ans_str],
                "answer_str": ans_str,
                "img_path": img_path
            })
            if len(self.samples) >= max_samples:
                break
                
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
        ])
        
    def __len__(self):
        return len(self.samples)
        
    def __getitem__(self, idx):
        item = self.samples[idx]
        img = Image.open(item["img_path"]).convert("RGB")
        img_t = self.transform(img)
        
        words = item["question"].lower().replace('?', '').replace(',', '').split()
        seq = [self.q_vocab.get(w, self.q_vocab["<UNK>"]) for w in words]
        if len(seq) == 0: seq = [self.q_vocab["<UNK>"]]
        seq_len = len(seq)
        
        max_len = 20
        if seq_len < max_len:
            seq = seq + [self.q_vocab["<PAD>"]] * (max_len - seq_len)
        else:
            seq = seq[:max_len]
            seq_len = max_len
            
        return img_t, torch.tensor(seq, dtype=torch.long), torch.tensor(seq_len, dtype=torch.long), torch.tensor(item["answer_id"], dtype=torch.long), item

def main():
    print("Loading validation data...")
    val_ds = RSVQA_TestDataset(split="val", max_samples=500)
    val_dl = DataLoader(val_ds, batch_size=1, shuffle=False, num_workers=0)
    
    with open('datasets/processed/rsvqa/q_vocab.json') as f:
        vocab_size = len(json.load(f))
    with open('datasets/processed/rsvqa/a_vocab.json') as f:
        num_answers = len(json.load(f))
        
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Using device: {device}")
    
    model = SimpleRSVQAModel(vocab_size=vocab_size, num_answers=num_answers).to(device)
    model.load_state_dict(torch.load("checkpoints/vqa_baseline.pth", map_location=device))
    model.eval()
    
    correct = 0
    total = 0
    
    os.makedirs('results/vqa', exist_ok=True)
    sample_results = []
    
    print("Evaluating...")
    t0 = time.time()
    for imgs, seqs, lens, ans, item in tqdm(val_dl):
        imgs, seqs, lens, ans = imgs.to(device), seqs.to(device), lens, ans.to(device)
        
        with torch.inference_mode():
            logits = model(imgs, seqs, lens)
            preds = logits.argmax(dim=1)
            probs = torch.softmax(logits, dim=1)
            conf = probs[0, preds[0]].item()
            
        correct += (preds == ans).sum().item()
        total += 1
        
        if total <= 10:
            sample_results.append({
                "image": item["img_path"][0],
                "question": item["question"][0],
                "true_answer": item["answer_str"][0],
                "predicted_answer": val_ds.idx_to_ans[preds[0].item()],
                "confidence": conf
            })
            
    eval_time = time.time() - t0
    acc = correct / total if total > 0 else 0
    
    out_data = {
        "model": "Custom-RSVQA-ResNet18-GRU",
        "dataset": "RSVQA-LR (val split, 500 samples)",
        "accuracy": acc,
        "inference_time_seconds": eval_time,
        "fps": total / eval_time,
        "samples": sample_results
    }
    
    with open('results/vqa/eval_metrics.json', 'w') as f:
        json.dump(out_data, f, indent=4)
        
    print(f"Accuracy: {acc:.4f}, FPS: {total / eval_time:.2f}")
    print("Saved results to results/vqa/eval_metrics.json")

if __name__ == '__main__':
    main()
