from src.tools.base import SpecialistTool
from src.agent.schemas import ToolRequest, ToolResult

class GroundingTool(SpecialistTool):
    name = "grounding"
    supported_workflows = ["grounding"]
    supported_modalities = ["optical"]
    model_name = "MockGroundingModel-DEMO"
    
    def can_handle(self, request: ToolRequest) -> bool:
        return len(request.images) == 1 and request.query is not None
        
    def execute(self, request: ToolRequest) -> ToolResult:
        return ToolResult(
            tool_name=self.name,
            success=True,
            data={
                "answer": f"[DEMO/FALLBACK MODE] Identified requested region '{request.query}'",
                "bounding_boxes": [[0.1, 0.1, 0.5, 0.5]], # Dummy coordinates
                "confidence": 0.5
            }
        )
