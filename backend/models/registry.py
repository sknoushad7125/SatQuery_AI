import os
from typing import Dict, List
from backend.tools.base import RemoteSensingTool

class ModelRegistry:
    def __init__(self):
        self._tools: Dict[str, RemoteSensingTool] = {}

    def register(self, tool: RemoteSensingTool):
        self._tools[tool.name] = tool

    def get_tool(self, name: str) -> RemoteSensingTool:
        return self._tools.get(name)

    def get_tools_for_task(self, task_type: str) -> List[RemoteSensingTool]:
        return [t for t in self._tools.values() if task_type in t.task_types]

registry = ModelRegistry()

def initialize_registry():
    use_mocks = os.environ.get("USE_MOCK_MODELS", "false").lower() == "true"
    
    if use_mocks:
        print("WARNING: Loading MOCK tools for development mode.")
        from backend.tools.vqa.mock_vqa import MockVQATool
        from backend.tools.grounding.mock_grounding import MockGroundingTool
        from backend.tools.change.mock_change import MockChangeTool
        from backend.tools.optical_sar.mock_fusion import MockOpticalSARTool
        
        registry.register(MockVQATool())
        registry.register(MockGroundingTool())
        registry.register(MockChangeTool())
        registry.register(MockOpticalSARTool())
    else:
        print("INFO: Loading REAL remote sensing models.")
        from backend.tools.vqa.real_vqa import RealVQATool
        from backend.tools.grounding.real_grounding import RealGroundingTool
        from backend.tools.change.real_change import RealSemanticChangeTool
        from backend.tools.optical_sar.real_fusion import RealDecisionFusionTool
        from backend.tools.classification.real_bigearthnet import RealBigEarthNetTool
        
        registry.register(RealVQATool())
        registry.register(RealGroundingTool())
        registry.register(RealSemanticChangeTool())
        registry.register(RealDecisionFusionTool())
        registry.register(RealBigEarthNetTool())
        
initialize_registry()
