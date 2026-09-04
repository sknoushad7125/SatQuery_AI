import os
import time
import json
import torch
import torch.nn as nn
import numpy as np
import io
import base64
import cv2
from PIL import Image
from typing import List
import albumentations as A
from albumentations.pytorch import ToTensorV2

from backend.tools.base import RemoteSensingTool
from backend.api.schemas.domain import ImageMetadata, ToolResult
from src.models.baseline import SiameseChangeDetector

class RealSemanticChangeTool(RemoteSensingTool):
    name = "RealSemanticChangeModel"
    version = "4.0.0"
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
            project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../.."))
            ckpt_path = os.path.join(project_root, "checkpoints", "best_bce_dice.pth")

            if not os.path.exists(ckpt_path):
                raise FileNotFoundError(f"Verified change-detection checkpoint missing at {ckpt_path}.")

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
        if mod1 != mod2: return False
        if mod1 not in self.modalities: return False
        return True

    def run(self, images: List[str], query: str, **kwargs) -> ToolResult:
        start_time = time.time()

        try:
            self._load_model()

            img_a = np.array(Image.open(images[0]).convert("RGB"))
            img_b = np.array(Image.open(images[1]).convert("RGB"))

            transformed = self.transform(image=img_a, image0=img_b)
            t_img_a = transformed["image"].unsqueeze(0).to(self.device)
            t_img_b = transformed["image0"].unsqueeze(0).to(self.device)

            with torch.no_grad():
                out = self.model(t_img_a, t_img_b)
                probs = torch.sigmoid(out)
                preds = (probs > 0.5).cpu().numpy()[0, 0]
                conf_map = probs[0, 0].cpu().numpy()

            changed_pixels = np.sum(preds)
            total_pixels = preds.size
            perc_change = (changed_pixels / total_pixels) * 100

            conf = float(np.mean(conf_map[preds > 0])) if changed_pixels > 0 else float(np.max(conf_map))

            method = "SiameseResNet18 Phase4"
            ckpt = "best_bce_dice.pth"
            text = f"Bi-temporal change analysis using {method}. Detected {perc_change:.2f}% change area."

            mask_img = Image.fromarray((preds * 255).astype(np.uint8))
            buffered = io.BytesIO()
            mask_img.save(buffered, format="PNG")
            mask_b64 = base64.b64encode(buffered.getvalue()).decode("utf-8")

            # Extract change crops
            change_crops = []
            img_h, img_w = img_a.shape[:2]
            contours, _ = cv2.findContours((preds * 255).astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            valid_contours = [c for c in contours if cv2.contourArea(c) >= 50]
            valid_contours.sort(key=cv2.contourArea, reverse=True)
            top_contours = valid_contours[:3]

            for i, c in enumerate(top_contours):
                x, y, w, h = cv2.boundingRect(c)
                pad_x = int(w * 0.2)
                pad_y = int(h * 0.2)
                x1 = max(0, x - pad_x)
                y1 = max(0, y - pad_y)
                x2 = min(img_w, x + w + pad_x)
                y2 = min(img_h, y + h + pad_y)

                crop_a = img_a[y1:y2, x1:x2]
                crop_b = img_b[y1:y2, x1:x2]

                # Convert to PIL to resize & save
                pil_a = Image.fromarray(crop_a)
                pil_b = Image.fromarray(crop_b)

                # Resize if max dimension > 256
                max_dim = max(pil_a.width, pil_a.height)
                if max_dim > 256:
                    scale = 256.0 / max_dim
                    new_w = int(pil_a.width * scale)
                    new_h = int(pil_a.height * scale)
                    pil_a = pil_a.resize((new_w, new_h), Image.Resampling.LANCZOS)
                    pil_b = pil_b.resize((new_w, new_h), Image.Resampling.LANCZOS)

                buf_a = io.BytesIO()
                buf_b = io.BytesIO()
                pil_a.save(buf_a, format="JPEG", quality=85)
                pil_b.save(buf_b, format="JPEG", quality=85)

                b64_a = base64.b64encode(buf_a.getvalue()).decode("utf-8")
                b64_b = base64.b64encode(buf_b.getvalue()).decode("utf-8")

                change_crops.append({
                    "region_id": i + 1,
                    "area_pixels": float(cv2.contourArea(c)),
                    "bbox": [x, y, w, h],
                    "before": b64_a,
                    "after": b64_b
                })

            return ToolResult(
                tool_name=self.name,
                model_name=method,
                model_type=self.execution_type,
                status="success",
                text=text,
                structured_data={"percentage_change": perc_change},
                confidence=conf,
                evidence={"percentage_change": perc_change, "change_mask_b64": mask_b64, "change_crops": change_crops},
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
