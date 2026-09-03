from abc import ABC, abstractmethod
from typing import List, Literal, Optional, Any, Dict
from backend.api.schemas.domain import ImageMetadata, ToolResult

class RemoteSensingTool(ABC):
    """
    Universal contract for all specialist models/tools.
    """
    name: str
    version: str
    task_types: List[str]
    modalities: List[str]
    execution_type: Literal["real", "mock"]

    @abstractmethod
    def validate(self, images: List[ImageMetadata], query: str) -> bool:
        """Validates if the inputs are suitable for this tool."""
        pass

    @abstractmethod
    def run(self, images: List[str], query: str, **kwargs) -> ToolResult:
        """
        Executes the tool.
        Returns a structured result:
        - text: natural language answer/description
        - spatial evidence: dict containing geometries, bounding boxes, or masks
        - confidence: float score
        - metadata: internal states or raw outputs
        - execution information: timing, provenance
        """
        pass
