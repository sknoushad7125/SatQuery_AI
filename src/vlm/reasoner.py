from typing import List
from src.agent.schemas import Evidence, InputConfiguration, SatQueryResponse, ExecutionTrace
from src.vlm.base import VLMProvider

class MockVLMProvider(VLMProvider):
    def generate(self, query: str, config: InputConfiguration, evidence: List[Evidence]) -> str:
        # Fallback implementation
        obs = [e.text_observations for e in evidence if e.text_observations]
        base_ans = " ".join(obs)
        if not base_ans:
            base_ans = "Processed evidence successfully."
        return f"Based on the evidence: {base_ans}"

class VLMReasoner:
    def __init__(self, provider: VLMProvider = None):
        self.provider = provider or MockVLMProvider()
        
    def answer(
        self,
        query: str,
        config: InputConfiguration,
        evidence: List[Evidence],
        trace: ExecutionTrace
    ) -> SatQueryResponse:
    
        ans = self.provider.generate(query, config, evidence)
        conf = sum(e.confidence for e in evidence) / len(evidence) if evidence else 0.0
        
        limitations = ["[DEMO/FALLBACK MODE] Simulated VLM response"] if isinstance(self.provider, MockVLMProvider) else []
        
        return SatQueryResponse(
            answer=ans,
            confidence=conf,
            workflow=trace.task,
            evidence=evidence,
            visual_outputs=[],
            execution_trace=trace,
            limitations=limitations
        )
