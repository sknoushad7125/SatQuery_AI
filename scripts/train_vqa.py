import json
import os
import time
import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from torchvision import transforms
from src.models.vqa import SimpleRSVQAModel
from tqdm import tqdm

class RSVQA_Dataset(Dataset):
    def __init__(self, split="train", max_samples=1000):
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
            
        # Filter active ones
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
        img = self.transform(img)
        
        words = item["question"].lower().replace('?', '').replace(',', '').split()
        seq = [self.q_vocab.get(w, self.q_vocab["<UNK>"]) for w in words]
        if len(seq) == 0: seq = [self.q_vocab["<UNK>"]]
        seq_len = len(seq)
        
        # Pad to fixed length (e.g., 20)
        max_len = 20
        if seq_len < max_len:
            seq = seq + [self.q_vocab["<PAD>"]] * (max_len - seq_len)
        else:
            seq = seq[:max_len]
            seq_len = max_len
            
        return img, torch.tensor(seq, dtype=torch.long), torch.tensor(seq_len, dtype=torch.long), torch.tensor(item["answer_id"], dtype=torch.long)

def main():
    print("Loading data...")
    train_ds = RSVQA_Dataset(split="train", max_samples=1000)
    train_dl = DataLoader(train_ds, batch_size=16, shuffle=True, num_workers=0)
    
    with open('datasets/processed/rsvqa/q_vocab.json') as f:
        vocab_size = len(json.load(f))
    with open('datasets/processed/rsvqa/a_vocab.json') as f:
        num_answers = len(json.load(f))
        
    print(f"Vocab size: {vocab_size}, Answers: {num_answers}")
    
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Using device: {device}")
    
    model = SimpleRSVQAModel(vocab_size=vocab_size, num_answers=num_answers).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    criterion = torch.nn.CrossEntropyLoss()
    
    model.train()
    for epoch in range(1):
        total_loss = 0
        correct = 0
        for imgs, seqs, lens, ans in tqdm(train_dl):
            imgs, seqs, lens, ans = imgs.to(device), seqs.to(device), lens, ans.to(device)
            
            optimizer.zero_grad()
            logits = model(imgs, seqs, lens)
            loss = criterion(logits, ans)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            preds = logits.argmax(dim=1)
            correct += (preds == ans).sum().item()
            
        print(f"Epoch {epoch+1} | Loss: {total_loss/len(train_dl):.4f} | Acc: {correct/len(train_ds):.4f}")
        
    os.makedirs('checkpoints', exist_ok=True)
    torch.save(model.state_dict(), 'checkpoints/vqa_baseline.pth')
    print("Saved to checkpoints/vqa_baseline.pth")

if __name__ == '__main__':
    main()
