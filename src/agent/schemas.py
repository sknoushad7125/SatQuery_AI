from pydantic import BaseModel, Field
from typing import List, Dict, Any, Optional
from enum import Enum

class Modality(str, Enum):
    OPTICAL = "optical"
    MULTISPECTRAL = "multispectral"
    SAR = "sar"
    UNKNOWN = "unknown"

class WorkflowType(str, Enum):
    SINGLE_VQA = "single_vqa"
    CAPTIONING = "captioning"
    GROUNDING = "grounding"
    CHANGE_ANALYSIS = "change_analysis"
    CHANGE_VQA = "change_vqa"
    OPTICAL_SAR_ANALYSIS = "optical_sar_analysis"

class ImageInput(BaseModel):
    filepath: str
    modality: Modality = Modality.UNKNOWN
    metadata: Dict[str, Any] = Field(default_factory=dict)

class InputConfiguration(BaseModel):
    number_of_images: int
    modality: Modality
    image_format: List[str]
    dimensions: List[List[int]]
    geospatial_metadata_available: bool
    temporal_pair: bool
    cross_modal_pair: bool

class QueryIntent(BaseModel):
    workflow: WorkflowType
    confidence: float
    required_tools: List[str]
    reason: str

class ToolRequest(BaseModel):
    tool_name: str
    images: List[ImageInput]
    query: Optional[str] = None
    parameters: Dict[str, Any] = Field(default_factory=dict)

class ToolResult(BaseModel):
    tool_name: str
    success: bool
    data: Dict[str, Any]
    error: Optional[str] = None
    visual_outputs: List[str] = Field(default_factory=dict)

class Evidence(BaseModel):
    source_tool: str
    model_name: str
    confidence: float
    spatial_regions: List[Dict[str, Any]] = Field(default_factory=list)
    mask_path: Optional[str] = None
    bounding_boxes: List[List[float]] = Field(default_factory=list)
    numeric_statistics: Dict[str, Any] = Field(default_factory=dict)
    text_observations: str

class ExecutionTrace(BaseModel):
    task: str
    input_configuration: Dict[str, Any]
    selected_tools: List[str]
    models: List[str]
    parameters: Dict[str, Any]
    outputs: List[str]
    confidence: float
    execution_time_seconds: float

class SatQueryResponse(BaseModel):
    answer: str
    confidence: float
    workflow: str
    evidence: List[Evidence]
    visual_outputs: List[str]
    execution_trace: ExecutionTrace
    limitations: List[str]
