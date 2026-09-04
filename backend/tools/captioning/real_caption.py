import time
import torch
from typing import List
from PIL import Image
from transformers import BlipProcessor, BlipForConditionalGeneration

from backend.tools.base import RemoteSensingTool
from backend.api.schemas.domain import ImageMetadata, ToolResult
import rasterio

class RealCaptioningTool(RemoteSensingTool):
    name = "Gurveer05_BLIP_RSICD"
    version = "1.0.0"
    task_types = ["single_captioning", "scene_description"]
    modalities = ["optical", "multispectral"]
    execution_type = "real"

    def __init__(self):
        self.model = None
        self.processor = None
        self.device = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")
        self.model_id = "Gurveer05/blip-image-captioning-base-rscid-finetuned"
        self.dataset = "RSICD"

    def _load_model(self):
        if self.model is None:
            t0 = time.time()
            self.processor = BlipProcessor.from_pretrained(self.model_id)
            self.model = BlipForConditionalGeneration.from_pretrained(self.model_id).to(self.device)
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
                error_message="Captioning requires exactly one image.",
                execution_time_ms=(time.time() - start_time) * 1000
            )

        try:
            self._load_model()
            img_path = images[0]

            # TIFF fallback
            if img_path.lower().endswith(('.tif', '.tiff')):
                with rasterio.open(img_path) as src:
                    # Just read first 3 bands for RGB
                    arr = src.read([1, 2, 3]) if src.count >= 3 else src.read()
                    # Reshape to HWC
                    import numpy as np
                    arr = np.transpose(arr, (1, 2, 0))
                    # Normalize to 0-255 if not already
                    if arr.max() <= 1.0 or arr.dtype == np.float32:
                        arr = (np.clip(arr, 0, 1) * 255).astype(np.uint8)

                    if arr.shape[2] == 1:
                        arr = np.repeat(arr, 3, axis=2)
                    img = Image.fromarray(arr).convert("RGB")
            else:
                img = Image.open(img_path).convert("RGB")

            inputs = self.processor(img, return_tensors="pt").to(self.device)

            with torch.no_grad():
                out = self.model.generate(**inputs, max_new_tokens=50)

            caption = self.processor.decode(out[0], skip_special_tokens=True)

            return ToolResult(
                tool_name=self.name,
                model_name=self.model_id,
                model_type=self.execution_type,
                status="success",
                text=caption,
                # NOT fabricating confidence.
                confidence=None,
                execution_time_ms=(time.time() - start_time) * 1000,
                metadata={
                    "dataset": self.dataset,
                    "device": self.device,
                    "query_used": query,
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
