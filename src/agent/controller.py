import time
from typing import List
from src.agent.schemas import ImageInput, SatQueryResponse, ToolRequest, ExecutionTrace
from src.agent.input_validator import InputValidator
from src.agent.intent_router import IntentRouter
from src.agent.tool_registry import ToolRegistry
from src.agent.evidence import EvidenceAggregator
from src.vlm.reasoner import VLMReasoner

# Tool imports
from src.tools.change_detection import ChangeDetectionTool
from src.tools.single_image_vqa import SingleImageVQATool
from src.tools.grounding import GroundingTool
from src.tools.captioning import CaptioningTool
from src.tools.optical_sar import OpticalSARAnalyzer

class SatQueryController:
    def __init__(self):
        self.validator = InputValidator()
        self.router = IntentRouter()
        self.registry = ToolRegistry()
        self.aggregator = EvidenceAggregator()
        self.reasoner = VLMReasoner()
        
        self.registry.register(ChangeDetectionTool())
        self.registry.register(SingleImageVQATool())
        self.registry.register(GroundingTool())
        self.registry.register(CaptioningTool())
        self.registry.register(OpticalSARAnalyzer())
        
    def process_query(self, images: List[ImageInput], query: str) -> SatQueryResponse:
        t0 = time.time()
        
        valid, msg, config = self.validator.validate(images)
        if not valid:
            raise ValueError(f"Input Validation Failed: {msg}")
            
        intent = self.router.route(query, config)
        
        tool_results = []
        selected_tools = []
        models_used = []
        
        for t_name in intent.required_tools:
            tool = self.registry.get_tool(t_name)
            if not tool:
                continue
            req = ToolRequest(tool_name=t_name, images=images, query=query)
            if tool.can_handle(req):
                selected_tools.append(t_name)
                models_used.append(tool.model_name)
                res = tool.execute(req)
                tool_results.append(res)
                
        tool_model_map = {t: m for t, m in zip(selected_tools, models_used)}
        evidences = self.aggregator.aggregate(tool_results, tool_model_map)
        
        trace = ExecutionTrace(
            task=intent.workflow.value,
            input_configuration=config.model_dump(),
            selected_tools=selected_tools,
            models=models_used,
            parameters={},
            outputs=[key for r in tool_results if r.success for key in r.data.keys()],
            confidence=intent.confidence,
            execution_time_seconds=time.time() - t0
        )
        
        response = self.reasoner.answer(query, config, evidences, trace)
        return response
