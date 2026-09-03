import os
import time
import torch
import numpy as np
from PIL import Image
from typing import List
import albumentations as A
from albumentations.pytorch import ToTensorV2

from backend.tools.base import RemoteSensingTool
from backend.api.schemas.domain import ImageMetadata, ToolResult
from src.models.baseline import SiameseChangeDetector

class RealSemanticChangeTool(RemoteSensingTool):
    name = "RealSemanticChangeModel"
    version = "4.0.0"  # Phase 4 integration
    task_types = ["bi_temporal_change", "change_vqa"]
    modalities = ["optical", "multispectral", "sar"]
    execution_type = "real"

    def __init__(self):
        self.model = None
        self.device = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")
        self.training_status = "trained_checkpoint_loaded"
        self.dataset = "LEVIR-CD"
        self.transform = None

    def _load_model(self):
        if self.model is None:
            # Resolve relative to project root
            project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../.."))
            ckpt_path = os.path.join(project_root, "checkpoints", "best_bce_dice.pth")
            
            if not os.path.exists(ckpt_path):
                raise FileNotFoundError(f"Verified change-detection checkpoint missing at {ckpt_path}. Refusing to fall back to DINOv2.")
                
            print(f"Loading Phase 4 Verified SiameseResNet18 from {ckpt_path}...")
            
            self.model = SiameseChangeDetector(encoder_name="resnet18", pretrained=False)
            self.model.load_state_dict(torch.load(ckpt_path, map_location=self.device))
            self.model.to(self.device)
            self.model.eval()
            
            self.transform = A.Compose([
                A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
                ToTensorV2(),
            ], additional_targets={'image0': 'image'})

    def validate(self, images: List[ImageMetadata], query: str) -> bool:
        if len(images) != 2: 
            return False
        
        mod1 = images[0].modality.lower()
        mod2 = images[1].modality.lower()
        
        if mod1 != mod2:
            print(f"Rejecting cross-modal change detection ({mod1} vs {mod2}).")
            return False
            
        if mod1 not in self.modalities:
            return False
            
        return True

    def run(self, images: List[str], query: str, **kwargs) -> ToolResult:
        start_time = time.time()
        
        try:
            self._load_model()
            
            # 1. Load images
            img_a = np.array(Image.open(images[0]).convert("RGB"))
            img_b = np.array(Image.open(images[1]).convert("RGB"))
            
            # 2. Preprocess consistently
            transformed = self.transform(image=img_a, image0=img_b)
            t_img_a = transformed["image"].unsqueeze(0).to(self.device)
            t_img_b = transformed["image0"].unsqueeze(0).to(self.device)
            
            # 3. Model inference
            with torch.no_grad():
                out = self.model(t_img_a, t_img_b)
                probs = torch.sigmoid(out)
                preds = (probs > 0.5).cpu().numpy()[0, 0]
                conf_map = probs[0, 0].cpu().numpy()
                
            changed_pixels = np.sum(preds)
            total_pixels = preds.size
            perc_change = (changed_pixels / total_pixels) * 100
            
            # Confidence
            conf = float(np.mean(conf_map[preds > 0])) if changed_pixels > 0 else float(np.max(conf_map))
            
            method = "SiameseResNet18 Phase4"
            ckpt = "best_bce_dice.pth"
            
            text = f"Bi-temporal change analysis using {method}. Detected {perc_change:.2f}% change area."
            
            return ToolResult(
                tool_name=self.name,
                model_name=method,
                model_type=self.execution_type,
                status="success",
                text=text,
                confidence=conf,
                evidence={"percentage_change": perc_change, "change_mask_data": preds.tolist()[:10]}, 
                execution_time_ms=(time.time() - start_time) * 1000,
                metadata={
                    "checkpoint": ckpt,
                    "dataset": self.dataset,
                    "status": self.training_status
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
