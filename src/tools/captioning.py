from src.tools.base import SpecialistTool
from src.agent.schemas import ToolRequest, ToolResult

class CaptioningTool(SpecialistTool):
    name = "captioning"
    supported_workflows = ["captioning"]
    supported_modalities = ["optical"]
    model_name = "MockCaptioningModel-DEMO"
    
    def can_handle(self, request: ToolRequest) -> bool:
        return len(request.images) == 1
        
    def execute(self, request: ToolRequest) -> ToolResult:
        return ToolResult(
            tool_name=self.name,
            success=True,
            data={
                "answer": "[DEMO/FALLBACK MODE] This is a remote sensing scene containing various geographical features.",
                "confidence": 0.5
            }
        )
