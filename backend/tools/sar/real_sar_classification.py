import os
import time
import torch
import numpy as np
import rasterio
from typing import List

import timm
from safetensors.torch import load_file
from huggingface_hub import hf_hub_download

from backend.tools.base import RemoteSensingTool
from backend.api.schemas.domain import ImageMetadata, ToolResult

BIGEARTHNET_CLASSES = [
    "Agro-forestry areas",
    "Arable land",
    "Beaches, dunes, sands",
    "Broad-leaved forest",
    "Coastal wetlands",
    "Complex cultivation patterns",
    "Coniferous forest",
    "Industrial or commercial units",
    "Inland waters",
    "Inland wetlands",
    "Land principally occupied by agriculture, with significant areas of natural vegetation",
    "Marine waters",
    "Mixed forest",
    "Moors, heathland and sclerophyllous vegetation",
    "Natural grassland and sparsely vegetated areas",
    "Pastures",
    "Permanent crops",
    "Transitional woodland, shrub",
    "Urban fabric"
]

class RealSARClassificationTool(RemoteSensingTool):
    name = "BigEarthNet_ResNet18_SAR"
    version = "1.0.0"
    task_types = ["single_sar_classification"]
    modalities = ["sar"]
    execution_type = "real"

    def __init__(self):
        self.model = None
        self.device = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")
        self.model_id = "BIFOLD-BigEarthNetv2-0/resnet18-s1-v0.2.0"
        self.dataset = "BigEarthNet-MM (Sentinel-1)"

    def _load_model(self):
        if self.model is None:
            t0 = time.time()
            weights_path = hf_hub_download(repo_id=self.model_id, filename="model.safetensors")
            state_dict = load_file(weights_path)

            # Strip custom wrapper prefix to load into standard timm
            new_state_dict = {}
            for k, v in state_dict.items():
                if k.startswith("model.vision_encoder."):
                    new_k = k.replace("model.vision_encoder.", "")
                    new_state_dict[new_k] = v
                else:
                    new_state_dict[k] = v

            self.model = timm.create_model('resnet18', pretrained=False, in_chans=2, num_classes=19)
            self.model.load_state_dict(new_state_dict, strict=False)
            self.model.to(self.device)
            self.model.eval()
            self.load_time_ms = (time.time() - t0) * 1000

    def validate(self, images: List[ImageMetadata], query: str) -> bool:
        if not images:
            return False
        return images[0].modality in self.modalities

    def run(self, images: List[str], query: str, **kwargs) -> ToolResult:
        start_time = time.time()

        if not images or len(images) != 1:
            return ToolResult(
                tool_name=self.name,
                model_name=self.model_id,
                model_type=self.execution_type,
                status="failure",
                error_message="SAR Classification requires exactly one SAR image.",
                execution_time_ms=(time.time() - start_time) * 1000
            )

        try:
            self._load_model()
            img_path = images[0]

            with rasterio.open(img_path) as src:
                # Expecting VV and VH bands. Read first 2 bands.
                arr = src.read([1, 2]) if src.count >= 2 else src.read()

            # Standard SEN12MS/BigEarthNet SAR dB clipping
            arr = arr.astype(np.float32)
            arr = np.clip(arr, -25.0, 0.0) / -25.0

            # If input is 1 band, duplicate to 2 for the model
            if arr.shape[0] == 1:
                arr = np.repeat(arr, 2, axis=0)

            # Resize to exactly 120x120 which BigEarthNet models expect by default,
            # though ResNet is fully convolutional before pooling, we'll interpolate for safety
            tensor = torch.from_numpy(arr).unsqueeze(0)
            tensor = torch.nn.functional.interpolate(tensor, size=(120, 120), mode='bilinear', align_corners=False)
            tensor = tensor.to(self.device)

            with torch.no_grad():
                logits = self.model(tensor)
                probs = torch.sigmoid(logits)[0] # Multi-label classification uses Sigmoid

            probs_np = probs.cpu().numpy()

            # Threshold predictions (e.g. > 0.5)
            predictions = {}
            for i, p in enumerate(probs_np):
                if p > 0.3:  # Lower threshold commonly used in BigEarthNet multi-label
                    predictions[BIGEARTHNET_CLASSES[i]] = float(p)

            if not predictions:
                # Fallback to top-1 if none pass threshold
                top_idx = np.argmax(probs_np)
                predictions[BIGEARTHNET_CLASSES[top_idx]] = float(probs_np[top_idx])

            pred_text = ", ".join([f"{k} ({v:.1%})" for k, v in sorted(predictions.items(), key=lambda item: item[1], reverse=True)])
            text = f"SAR Scene Classification detected the following land cover classes: {pred_text}"

            return ToolResult(
                tool_name=self.name,
                model_name=self.model_id,
                model_type=self.execution_type,
                status="success",
                text=text,
                structured_data={"predictions": predictions},
                confidence=None,
                execution_time_ms=(time.time() - start_time) * 1000,
                metadata={
                    "dataset": self.dataset,
                    "device": self.device,
                    "query_used": query,
                    "input_bands": 2,
                    "preprocessing": "clip [-25, 0] then normalize by -25",
                    "load_time_ms": getattr(self, "load_time_ms", 0)
                }
            )
        except Exception as e:
            return ToolResult(
                tool_name=self.name,
                model_name=self.model_id,
                model_type=self.execution_type,
                status="failure",
                error_message=str(e),
                execution_time_ms=(time.time() - start_time) * 1000
            )
