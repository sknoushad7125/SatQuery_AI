from abc import ABC, abstractmethod
from typing import Dict, Any

class EvaluationAdapter(ABC):
    name: str

    @abstractmethod
    def load_dataset(self) -> Any:
        pass

    @abstractmethod
    def evaluate(self, model_registry, limit: int = 100) -> Dict[str, float]:
        pass
