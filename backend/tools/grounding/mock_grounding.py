import time
from backend.tools.base import RemoteSensingTool
from backend.api.schemas.domain import ImageMetadata, ToolResult

class MockGroundingTool(RemoteSensingTool):
    name = "Mock_Grounding"
    version = "1.0.0"
    task_types = ["grounding"]
    modalities = ["optical", "multispectral", "sar"]
    execution_type = "mock"

    def validate(self, images: list[ImageMetadata], query: str) -> bool:
        return len(images) == 1

    def run(self, images: list[str], query: str, **kwargs) -> ToolResult:
        time.sleep(0.5)
        # Mock a bounding box [xmin, ymin, xmax, ymax]
        return ToolResult(
            tool_name=self.name,
            model_name="MockGroundingModel",
            model_type=self.execution_type,
            status="success",
            text=f"[MOCK] Grounded the region based on query: '{query}'",
            evidence={
                "type": "bounding_box",
                "coordinates": [0.2, 0.3, 0.4, 0.5]
            },
            confidence=0.90,
            execution_time_ms=500
        )
