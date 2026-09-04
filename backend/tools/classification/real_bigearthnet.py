import os
import time
import torch
import numpy as np
from PIL import Image
from typing import List
from transformers import AutoImageProcessor, AutoModelForImageClassification
from peft import PeftModel
from backend.tools.base import RemoteSensingTool
from backend.api.schemas.domain import ImageMetadata, ToolResult

# Original 43 BigEarthNet labels (subset for demonstration)
BIGEARTHNET_LABELS = [
    "Continuous urban fabric", "Discontinuous urban fabric", "Industrial or commercial units",
    "Road and rail networks", "Arable land", "Permanent crops", "Pastures", "Complex cultivation patterns",
    "Land principally occupied by agriculture", "Agro-forestry areas", "Broad-leaved forest",
    "Coniferous forest", "Mixed forest", "Natural grassland", "Moors and heathland",
    "Transitional woodland/shrub", "Beaches, dunes, sands", "Bare rock", "Sparsely vegetated areas",
    "Burnt areas", "Inland marshes", "Peat bogs", "Salt marshes", "Salines", "Intertidal flats",
    "Water courses", "Water bodies", "Coastal lagoons", "Estuaries", "Sea and ocean"
] # Truncated to 30 for simplicity, in a real scenario we'd map all 43 or 19.

class RealBigEarthNetTool(RemoteSensingTool):
    name = "BigEarthNetClassifier"
    version = "1.0.0"
    task_types = ["captioning", "scene_classification", "multi_tool"]
    modalities = ["optical", "multispectral"]
    execution_type = "real"

    def __init__(self):
        self.model = None
        self.processor = None
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

    def _load_model(self):
        if self.model is None:
            print("Loading Base ViT for BigEarthNet...")
            base_model_name = "google/vit-base-patch16-224-in21k"
            self.processor = AutoImageProcessor.from_pretrained(base_model_name)

            # Create a model with the right number of labels
            base_model = AutoModelForImageClassification.from_pretrained(
                base_model_name,
                num_labels=len(BIGEARTHNET_LABELS),
                ignore_mismatched_sizes=True
            ).to(self.device)

            lora_path = "/app/training/checkpoints/bigearthnet_lora"
            # Fallback for local testing
            if not os.path.exists(lora_path):
                lora_path = os.path.join(os.getcwd(), "training", "checkpoints", "bigearthnet_lora")

            if os.path.exists(lora_path):
                print(f"Loading LoRA adapted weights from {lora_path}")
                try:
                    self.model = PeftModel.from_pretrained(base_model, lora_path).to(self.device)
                except Exception as e:
                    print(f"Error loading LoRA weights, using base model: {e}")
                    self.model = base_model
            else:
                print("No LoRA weights found. Using base model (Classification will be random until trained).")
                self.model = base_model

            self.model.eval()

    def validate(self, images: List[ImageMetadata], query: str) -> bool:
        if not images:
            return False
        return images[0].modality in self.modalities

    def run(self, images: List[str], query: str, **kwargs) -> ToolResult:
        start_time = time.time()
        self._load_model()

        try:
            raw_image = Image.open(images[0]).convert('RGB')
            inputs = self.processor(raw_image, return_tensors="pt").to(self.device)

            with torch.no_grad():
                outputs = self.model(**inputs)
                logits = outputs.logits
                probs = torch.sigmoid(logits)[0].cpu().numpy()

            # Get classes with probability > 0.5 (or top 3 if none)
            predicted_indices = np.where(probs > 0.5)[0]
            if len(predicted_indices) == 0:
                predicted_indices = np.argsort(probs)[-3:]

            predicted_classes = [BIGEARTHNET_LABELS[i] for i in predicted_indices if i < len(BIGEARTHNET_LABELS)]
            avg_conf = float(np.mean(probs[predicted_indices])) if len(predicted_indices) > 0 else 0.0

            text = f"Scene classified using BigEarthNet classes. Dominant land-cover: {', '.join(predicted_classes)}."

            return ToolResult(
                tool_name=self.name,
                model_name="ViT-LoRA-BigEarthNet",
                model_type=self.execution_type,
                status="success",
                text=text,
                confidence=avg_conf,
                structured_data={"predicted_classes": predicted_classes, "probabilities": probs.tolist()},
                metadata={"adapter_loaded": isinstance(self.model, PeftModel)},
                execution_time_ms=(time.time() - start_time) * 1000
            )
        except Exception as e:
            return ToolResult(
                tool_name=self.name,
                model_name="ViT-LoRA-BigEarthNet",
                model_type=self.execution_type,
                status="failure",
                error_message=str(e),
                execution_time_ms=(time.time() - start_time) * 1000
            )
