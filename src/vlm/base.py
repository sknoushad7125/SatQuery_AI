from abc import ABC, abstractmethod
from typing import List
from src.agent.schemas import Evidence, InputConfiguration, SatQueryResponse

class VLMProvider(ABC):
    @abstractmethod
    def generate(self, query: str, config: InputConfiguration, evidence: List[Evidence]) -> str:
        pass
