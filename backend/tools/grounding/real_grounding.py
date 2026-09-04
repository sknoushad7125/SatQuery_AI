import os
import time
import torch
from PIL import Image
from typing import List
from backend.tools.base import RemoteSensingTool
from backend.api.schemas.domain import ImageMetadata, ToolResult

class RealGroundingTool(RemoteSensingTool):
    name = "RealGroundingModel"
    version = "2.0.0"
    task_types = ["grounding"]
    modalities = ["optical", "multispectral"]
    execution_type = "real"

    def __init__(self):
        self.processor = None
        self.model = None
        self.rs_adapted = False
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

    def _load_model(self):
        if self.model is None:
            print("Loading OWL-ViT for Grounding...")
            from transformers import OwlViTProcessor, OwlViTForObjectDetection
            from peft import PeftModel

            self.processor = OwlViTProcessor.from_pretrained("google/owlvit-base-patch32")
            base_model = OwlViTForObjectDetection.from_pretrained("google/owlvit-base-patch32").to(self.device)

            # ATTEMPT TO LOAD RS-ADAPTED WEIGHTS (VRSBench LoRA)
            lora_path = "/app/training/checkpoints/vrsbench_lora"
            if not os.path.exists(lora_path):
                lora_path = os.path.join(os.getcwd(), "training", "checkpoints", "vrsbench_lora")

            if os.path.exists(lora_path):
                print(f"Loading VRSBench-adapted weights from {lora_path}")
                try:
                    self.model = PeftModel.from_pretrained(base_model, lora_path).to(self.device)
                    self.rs_adapted = True
                except Exception as e:
                    print(f"Failed to load RS-adapted weights: {e}")
                    self.model = base_model
            else:
                print("VRSBench-adapted weights not found. Using zero-shot base model.")
                self.model = base_model

            self.model.eval()

    def validate(self, images: List[ImageMetadata], query: str) -> bool:
        if not images: return False
        return images[0].modality in self.modalities

    def run(self, images: List[str], query: str, **kwargs) -> ToolResult:
        start_time = time.time()
        self._load_model()

        try:
            raw_image = Image.open(images[0]).convert('RGB')

            target_obj = query.lower().replace("highlight the", "").replace("locate the", "").replace("find the", "").replace("highlight", "").strip()
            if not target_obj: target_obj = "object"

            texts = [[f"a photo of a {target_obj}"]]
            inputs = self.processor(text=texts, images=raw_image, return_tensors="pt").to(self.device)

            with torch.no_grad():
                outputs = self.model(**inputs)

            target_sizes = torch.Tensor([raw_image.size[::-1]]).to(self.device)
            results = self.processor.post_process_object_detection(outputs=outputs, target_sizes=target_sizes, threshold=0.1)

            i = 0
            text_labels = texts[i]
            boxes, scores, labels = results[i]["boxes"], results[i]["scores"], results[i]["labels"]

            detected = []
            max_score = 0.0
            for box, score, label in zip(boxes, scores, labels):
                box = [round(i, 2) for i in box.tolist()]
                score = round(score.item(), 3)
                if score > max_score: max_score = score
                detected.append({"box": box, "score": score, "label": text_labels[label.item()]})

            detected = sorted(detected, key=lambda x: x["score"], reverse=True)[:3]

            if not detected:
                text = f"Could not confidently locate '{target_obj}' in the image."
                conf = 0.0
            else:
                text = f"Successfully grounded '{target_obj}'. Found {len(detected)} region(s)."
                conf = float(max_score)

            return ToolResult(
                tool_name=self.name,
                model_name="Zero-shot OWL-ViT baseline",
                model_type=self.execution_type,
                status="success",
                text=text,
                confidence=conf,
                evidence={"bounding_boxes": detected},
                execution_time_ms=(time.time() - start_time) * 1000
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
