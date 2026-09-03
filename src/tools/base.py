from abc import ABC, abstractmethod
from typing import List
from src.agent.schemas import ToolRequest, ToolResult

class SpecialistTool(ABC):
    name: str
    supported_workflows: List[str]
    supported_modalities: List[str]
    model_name: str
    
    @abstractmethod
    def can_handle(self, request: ToolRequest) -> bool:
        pass
        
    @abstractmethod
    def execute(self, request: ToolRequest) -> ToolResult:
        pass
