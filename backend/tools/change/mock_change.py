import time
from backend.tools.base import RemoteSensingTool
from backend.api.schemas.domain import ImageMetadata, ToolResult

class MockChangeTool(RemoteSensingTool):
    name = "Mock_ChangeDetection"
    version = "1.0.0"
    task_types = ["bi_temporal_change", "change_vqa"]
    modalities = ["optical", "multispectral", "sar"]
    execution_type = "mock"

    def validate(self, images: list[ImageMetadata], query: str) -> bool:
        return len(images) == 2

    def run(self, images: list[str], query: str, **kwargs) -> ToolResult:
        time.sleep(0.8)
        return ToolResult(
            tool_name=self.name,
            model_name="MockChangeModel",
            model_type=self.execution_type,
            status="success",
            text="[MOCK] Built-up area increased in the northern portion of the scene. (+7.3 percentage points)",
            structured_data={
                "change_type": "built_up_expansion",
                "area_change_pct": 7.3
            },
            evidence={
                "type": "change_map",
                "url": "/mock_assets/change_mask.png"
            },
            confidence=0.88,
            execution_time_ms=800
        )
