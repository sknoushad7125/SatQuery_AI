import os
import time
import json
import torch
from typing import List
from PIL import Image
from torchvision import transforms

from backend.tools.base import RemoteSensingTool
from backend.api.schemas.domain import ImageMetadata, ToolResult
from src.models.vqa import SimpleRSVQAModel

class RealVQATool(RemoteSensingTool):
    name = "RealRSVQAModel"
    version = "4.0.1"  # Phase 4 integration (Validated)
    task_types = ["single_vqa"]  # Removed captioning
    modalities = ["optical", "multispectral"]
    execution_type = "real"

    def __init__(self):
        self.model = None
        self.device = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")
        self.training_status = "trained_checkpoint_loaded"
        self.dataset = "RSVQA"
        self.q_vocab = None
        self.idx_to_ans = None
        self.transform = None

    def _load_model(self):
        if self.model is None:
            project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../.."))
            ckpt_path = os.path.join(project_root, "checkpoints", "vqa_baseline.pth")
            q_vocab_path = os.path.join(project_root, "datasets", "processed", "rsvqa", "q_vocab.json")
            a_vocab_path = os.path.join(project_root, "datasets", "processed", "rsvqa", "a_vocab.json")

            if not os.path.exists(ckpt_path):
                raise FileNotFoundError(f"Verified VQA checkpoint missing at {ckpt_path}. Refusing to fall back to BLIP.")

            if not os.path.exists(q_vocab_path) or not os.path.exists(a_vocab_path):
                raise FileNotFoundError("RSVQA vocabulary files missing. Cannot initialize Phase 4 VQA model.")

            with open(q_vocab_path) as f:
                self.q_vocab = json.load(f)
            with open(a_vocab_path) as f:
                a_vocab_raw = json.load(f)
                self.idx_to_ans = {v: k for k, v in a_vocab_raw.items()}
                num_answers = len(a_vocab_raw)

            self.model = SimpleRSVQAModel(vocab_size=len(self.q_vocab), num_answers=num_answers).to(self.device)
            self.model.load_state_dict(torch.load(ckpt_path, map_location=self.device))
            self.model.eval()

            self.transform = transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
            ])

    def validate(self, images: List[ImageMetadata], query: str) -> bool:
        if not images: return False
        return images[0].modality in self.modalities

    def run(self, images: List[str], query: str, **kwargs) -> ToolResult:
        start_time = time.time()

        if not images or len(images) != 1:
            return ToolResult(
                tool_name=self.name,
                model_name=self.name,
                model_type=self.execution_type,
                status="failure",
                error_message="VQA requires exactly one image.",
                execution_time_ms=(time.time() - start_time) * 1000
            )

        try:
            # Reject unsupported queries immediately
            if "describe" in query.lower() or "caption" in query.lower():
                return ToolResult(
                    tool_name=self.name,
                    model_name=self.name,
                    model_type=self.execution_type,
                    status="failure",
                    error_message="Unsupported task: Phase 4 model does not support image captioning/description. Please ask a specific question.",
                    execution_time_ms=(time.time() - start_time) * 1000
                )

            self._load_model()

            # Load and preprocess image
            img = Image.open(images[0]).convert("RGB")
            img_t = self.transform(img).unsqueeze(0).to(self.device)

            # Preprocess text (Exact Phase 4 preprocessing)
            words = query.lower().replace('?', '').replace(',', '').split()
            seq = [self.q_vocab.get(w, self.q_vocab.get("<UNK>", 0)) for w in words]
            if len(seq) == 0: seq = [self.q_vocab.get("<UNK>", 0)]

            max_len = 20
            seq_len = len(seq)
            if seq_len < max_len:
                seq = seq + [self.q_vocab.get("<PAD>", 0)] * (max_len - seq_len)
            else:
                seq = seq[:max_len]
                seq_len = max_len

            seq_t = torch.tensor([seq], dtype=torch.long).to(self.device)
            len_t = torch.tensor([seq_len], dtype=torch.long)

            with torch.no_grad():
                logits = self.model(img_t, seq_t, len_t)
                probs = torch.softmax(logits, dim=1)
                conf, pred = torch.max(probs, dim=1)

            ans_str = self.idx_to_ans.get(pred.item(), "Unknown")

            method = "Custom-RSVQA-ResNet18-GRU"
            ckpt = "vqa_baseline.pth"

            return ToolResult(
                tool_name=self.name,
                model_name=method,
                model_type=self.execution_type,
                status="success",
                text=ans_str,
                confidence=conf.item(),
                execution_time_ms=(time.time() - start_time) * 1000,
                metadata={
                    "checkpoint": ckpt,
                    "dataset": self.dataset,
                    "status": self.training_status,
                    "query_used": query
                }
            )
        except Exception as e:
            return ToolResult(
                tool_name=self.name,
                model_name=self.name,
                model_type=self.execution_type,
                status="failure",
                error_message=str(e),
                execution_time_ms=(time.time() - start_time) * 1000
            )
