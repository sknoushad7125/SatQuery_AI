import json
import time
import torch
from torchvision import transforms
from PIL import Image
from src.tools.base import SpecialistTool
from src.agent.schemas import ToolRequest, ToolResult
from src.models.vqa import SimpleRSVQAModel

class SingleImageVQATool(SpecialistTool):
    name = "single_image_vqa"
    supported_workflows = ["single_vqa"]
    supported_modalities = ["optical"]
    model_name = "Custom-RSVQA-ResNet18-GRU"
    
    def __init__(self, checkpoint_path: str = "checkpoints/vqa_baseline.pth"):
        with open('datasets/processed/rsvqa/q_vocab.json') as f:
            self.q_vocab = json.load(f)
        with open('datasets/processed/rsvqa/a_vocab.json') as f:
            a_vocab_raw = json.load(f)
            self.idx_to_ans = {v: k for k, v in a_vocab_raw.items()}
            self.num_answers = len(a_vocab_raw)
            
        self.device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
        self.model = SimpleRSVQAModel(vocab_size=len(self.q_vocab), num_answers=self.num_answers).to(self.device)
        self.model.load_state_dict(torch.load(checkpoint_path, map_location=self.device))
        self.model.eval()
        
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
        ])
        
    def can_handle(self, request: ToolRequest) -> bool:
        return len(request.images) == 1 and request.query is not None
        
    def execute(self, request: ToolRequest) -> ToolResult:
        t0 = time.time()
        try:
            img = Image.open(request.images[0].filepath).convert("RGB")
            img_t = self.transform(img).unsqueeze(0).to(self.device)
            
            words = request.query.lower().replace('?', '').replace(',', '').split()
            seq = [self.q_vocab.get(w, self.q_vocab["<UNK>"]) for w in words]
            if len(seq) == 0: seq = [self.q_vocab["<UNK>"]]
            
            max_len = 20
            seq_len = len(seq)
            if seq_len < max_len:
                seq = seq + [self.q_vocab["<PAD>"]] * (max_len - seq_len)
            else:
                seq = seq[:max_len]
                seq_len = max_len
                
            seq_t = torch.tensor([seq], dtype=torch.long).to(self.device)
            len_t = torch.tensor([seq_len], dtype=torch.long)
            
            with torch.inference_mode():
                logits = self.model(img_t, seq_t, len_t)
                probs = torch.softmax(logits, dim=1)
                conf, pred = torch.max(probs, dim=1)
                
            ans_str = self.idx_to_ans.get(pred.item(), "Unknown")
            
            return ToolResult(
                tool_name=self.name,
                success=True,
                data={
                    "answer": ans_str,
                    "confidence": conf.item()
                }
            )
        except Exception as e:
            return ToolResult(tool_name=self.name, success=False, data={}, error=str(e))
