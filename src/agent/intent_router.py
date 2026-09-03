from src.agent.schemas import QueryIntent, WorkflowType, InputConfiguration

class IntentRouter:
    def route(self, query: str, config: InputConfiguration) -> QueryIntent:
        query_lower = query.lower() if query else ""
        
        if config.cross_modal_pair:
            return QueryIntent(
                workflow=WorkflowType.OPTICAL_SAR_ANALYSIS,
                confidence=1.0,
                required_tools=["optical_sar_analyzer"],
                reason="Cross-modal pair provided."
            )
            
        if config.temporal_pair:
            if "what" in query_lower and "change" not in query_lower and "?" in query_lower:
                workflow = WorkflowType.CHANGE_VQA
            else:
                workflow = WorkflowType.CHANGE_ANALYSIS
            return QueryIntent(
                workflow=workflow,
                confidence=0.9,
                required_tools=["change_detector"],
                reason="Temporal pair provided with change keyword/context."
            )
            
        # Single image
        if "describe" in query_lower or "caption" in query_lower:
            return QueryIntent(
                workflow=WorkflowType.CAPTIONING,
                confidence=0.9,
                required_tools=["captioning"],
                reason="Keyword 'describe' detected."
            )
        elif "highlight" in query_lower or "locate" in query_lower or "where" in query_lower:
            return QueryIntent(
                workflow=WorkflowType.GROUNDING,
                confidence=0.9,
                required_tools=["grounding"],
                reason="Grounding keywords detected."
            )
        else:
            return QueryIntent(
                workflow=WorkflowType.SINGLE_VQA,
                confidence=0.8,
                required_tools=["single_image_vqa"],
                reason="Defaulting to single image VQA."
            )
