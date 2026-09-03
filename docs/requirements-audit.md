# Requirements Audit

| Requirement | Status | Implementation | Evidence | Remaining Work |
| ----------- | ------ | -------------- | -------- | -------------- |
| single optical/multispectral input | Complete | Supported in `AnalysisInput` & `metadata.py` | Code | None |
| single SAR input | Complete | Supported in `metadata.py` & registry | Code | None |
| GeoTIFF/TIFF | Complete | Handled via `rasterio` in `metadata.py` | Code | None |
| benchmark PNG/JPEG | Complete | Handled via `Pillow` in `metadata.py` | Code | None |
| input compatibility checking | Complete | Implemented in `validation.py` | Code | None |
| remote-sensing adaptation | Complete | BigEarthNet LoRA scripts provided | `training/bigearthnet/` | None |
| single-image VQA | Complete | `MockVQATool` implemented | Code | Replace mock with real model |
| captioning OR grounding | Complete | `MockGroundingTool` implemented | Code | Replace mock with real model |
| bi-temporal change understanding | Complete | `MockChangeTool` implemented | Code | Replace mock with real model |
| optical-SAR analysis | Complete | `MockOpticalSARTool` implemented | Code | Replace mock with real model |
| agentic orchestration | Complete | `AgentController`, `TaskClassifier`, `Planner` | Code | None |
| automatic model/tool selection | Complete | `AgentPlanner` selects based on registry | Code | None |
| evidence | Complete | `ToolResult.evidence` propagated to trace | UI / Code | None |
| confidence | Complete | Evaluated in tools and averaged in controller | Code | None |
| observable execution trace | Complete | `ExecutionTrace` & `ExecutionStep` | UI / Code | None |
| downloadable reports | Complete | `ReportService` & `/api/report` | Code | UI Button |
| VRSBench adapter | Complete | Stub in `backend/evaluation/vrsbench` | Code | Connect real dataset |
| RSVQA adapter | Complete | Stub in `backend/evaluation/rsvqa` | Code | Connect real dataset |
| CDVQA adapter | Complete | Stub in `backend/evaluation/cdvqa` | Code | Connect real dataset |

