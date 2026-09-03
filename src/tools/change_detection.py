import os
from typing import List
from src.tools.base import SpecialistTool
from src.agent.schemas import ToolRequest, ToolResult
from src.interface import ChangeDetector

class ChangeDetectionTool(SpecialistTool):
    name = "change_detector"
    supported_workflows = ["change_analysis", "change_vqa"]
    supported_modalities = ["optical"]
    model_name = "SiameseResNet18-FPN"
    
    def __init__(self, checkpoint_path: str = "checkpoints/best_baseline.pth"):
        if os.path.exists(checkpoint_path):
            self.detector = ChangeDetector(checkpoint_path)
        else:
            self.detector = None # Fallback handling
            
    def can_handle(self, request: ToolRequest) -> bool:
        return len(request.images) == 2
        
    def execute(self, request: ToolRequest) -> ToolResult:
        if not self.detector:
            return ToolResult(
                tool_name=self.name,
                success=False,
                data={},
                error="ChangeDetector checkpoint not found. Ensure Phase 2 training completes."
            )
            
        try:
            res = self.detector.predict(request.images[0].filepath, request.images[1].filepath)
            
            return ToolResult(
                tool_name=self.name,
                success=True,
                data={
                    "changed": res["summary_features"]["changed_pixel_count"] > 0,
                    "confidence": res["confidence"],
                    "changed_area_percent": res["summary_features"]["changed_ratio"] * 100,
                    "bounding_boxes": res["change_regions"],
                    "statistics": res["summary_features"]
                },
                visual_outputs=[] # In a real scenario, save the mask and return path
            )
        except Exception as e:
            return ToolResult(tool_name=self.name, success=False, data={}, error=str(e))
