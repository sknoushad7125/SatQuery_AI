from typing import List
from backend.api.schemas.domain import AnalysisTask
from backend.tools.base import RemoteSensingTool
from backend.models.registry import registry

class AgentPlanner:
    def plan(self, task: AnalysisTask) -> List[RemoteSensingTool]:
        """Selects the tools to execute for the given task."""

        # We can perform multi-tool composition here!
        if task.task_type == "captioning":
            # Use BigEarthNet for RS-specific labels, and VQA for natural language
            ben = registry.get_tool("BigEarthNetClassifier")
            vqa = registry.get_tool("RealRSVQAModel")
            return [t for t in [ben, vqa] if t]

        elif task.task_type == "single_vqa":
            return [registry.get_tool("RealRSVQAModel")]

        elif task.task_type == "grounding":
            return [registry.get_tool("RealGroundingModel")]

        elif task.task_type in ["bi_temporal_change", "change_vqa"]:
            return [registry.get_tool("RealSemanticChangeModel")]

        elif task.task_type == "optical_sar_analysis":
            return [registry.get_tool("FeatureLevelOpticalSARFusion")]

        return []
