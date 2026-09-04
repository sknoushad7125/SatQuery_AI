from pydantic import BaseModel, Field
from typing import List, Optional, Literal, Dict, Any
from datetime import datetime

class ImageMetadata(BaseModel):
    filename: str
    format: str
    width: int
    height: int
    bands: int
    modality: Literal["optical", "multispectral", "sar", "unknown"]
    crs: Optional[str] = None
    transform: Optional[List[float]] = None
    bounds: Optional[List[float]] = None
    acquisition_time: Optional[datetime] = None
    resolution: Optional[float] = None
    georeferenced: bool = False
    checksum: Optional[str] = None

class AnalysisInput(BaseModel):
    input_type: Literal["single", "optical_sar_pair", "temporal_pair"]
    images: List[ImageMetadata]

class AnalysisQuery(BaseModel):
    text: str

class AnalysisTask(BaseModel):
    task_type: Literal[
        "single_vqa",
        "captioning",
        "single_sar_classification",
        "grounding",
        "bi_temporal_change",
        "change_vqa",
        "optical_sar_analysis",
        "multi_tool",
        "unsupported"
    ]

class ToolResult(BaseModel):
    tool_name: str
    model_name: str
    model_type: Literal["real", "mock"]
    status: Literal["success", "failure"]
    text: Optional[str] = None
    structured_data: Optional[Dict[str, Any]] = None
    evidence: Optional[Dict[str, Any]] = None
    confidence: Optional[float] = None
    metadata: Optional[Dict[str, Any]] = None
    error_message: Optional[str] = None

    execution_time_ms: Optional[float] = None

class ExecutionStep(BaseModel):
    step_name: str
    description: str
    status: Literal["pending", "running", "success", "failure"]
    tool_results: Optional[List[ToolResult]] = None
    error_message: Optional[str] = None
    timestamp: datetime = Field(default_factory=datetime.utcnow)

class ExecutionTrace(BaseModel):
    trace_id: str
    task: AnalysisTask
    steps: List[ExecutionStep]
    final_result: Optional[str] = None
    final_confidence: Optional[float] = None
    final_evidence: Optional[Dict[str, Any]] = None
    completed_at: Optional[datetime] = None
