import time
from backend.tools.base import RemoteSensingTool
from backend.api.schemas.domain import ImageMetadata, ToolResult

class MockOpticalSARTool(RemoteSensingTool):
    name = "Mock_OpticalSAR_Fusion"
    version = "1.0.0"
    task_types = ["optical_sar_analysis"]
    modalities = ["optical", "multispectral", "sar"]
    execution_type = "mock"

    def validate(self, images: list[ImageMetadata], query: str) -> bool:
        return len(images) == 2

    def run(self, images: list[str], query: str, **kwargs) -> ToolResult:
        time.sleep(1.0)
        return ToolResult(
            tool_name=self.name,
            model_name="MockFusionModel",
            model_type=self.execution_type,
            status="success",
            text="[MOCK] Fused optical and SAR features. Water bodies identified with high confidence from SAR, vegetation identified from optical.",
            structured_data={
                "optical": {"vegetation_prob": 0.92},
                "sar": {"water_prob": 0.95}
            },
            confidence=0.92,
            execution_time_ms=1000
        )
