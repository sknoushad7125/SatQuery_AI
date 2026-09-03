from typing import Dict, Optional
from src.tools.base import SpecialistTool

class ToolRegistry:
    def __init__(self):
        self._tools: Dict[str, SpecialistTool] = {}
        
    def register(self, tool: SpecialistTool):
        self._tools[tool.name] = tool
        
    def get_tool(self, name: str) -> Optional[SpecialistTool]:
        return self._tools.get(name)
        
    def get_all_tools(self) -> Dict[str, SpecialistTool]:
        return self._tools
