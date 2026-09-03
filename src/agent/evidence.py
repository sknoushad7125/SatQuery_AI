from typing import List
from src.agent.schemas import ToolResult, Evidence

class EvidenceAggregator:
    def aggregate(self, results: List[ToolResult], tool_model_map: dict) -> List[Evidence]:
        evidences = []
        for res in results:
            if not res.success: continue
            
            ev = Evidence(
                source_tool=res.tool_name,
                model_name=tool_model_map.get(res.tool_name, "unknown"),
                confidence=res.data.get("confidence", 0.0),
                spatial_regions=res.data.get("regions", []),
                mask_path=res.data.get("mask_path"),
                bounding_boxes=res.data.get("bounding_boxes", []),
                numeric_statistics=res.data.get("statistics", {}),
                text_observations=res.data.get("answer") or f"Execution complete. Changed: {res.data.get('changed', 'N/A')}"
            )
            evidences.append(ev)
            
        return evidences
