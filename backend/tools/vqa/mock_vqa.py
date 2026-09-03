import time
from backend.tools.base import RemoteSensingTool
from backend.api.schemas.domain import ImageMetadata, ToolResult

class MockVQATool(RemoteSensingTool):
    name = "Mock_RSVQA"
    version = "1.0.0"
    task_types = ["single_vqa"]
    modalities = ["optical", "multispectral", "sar"]
    execution_type = "mock"

    def validate(self, images: list[ImageMetadata], query: str) -> bool:
        return len(images) == 1

    def run(self, images: list[str], query: str, **kwargs) -> ToolResult:
        time.sleep(0.5) # Simulate inference
        return ToolResult(
            tool_name=self.name,
            model_name="MockVQAModel",
            model_type=self.execution_type,
            status="success",
            text=f"[MOCK] Detected query about: '{query}'. The scene appears to be predominantly agricultural with some residential areas.",
            confidence=0.85,
            execution_time_ms=500
        )
