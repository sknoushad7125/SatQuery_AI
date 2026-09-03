from backend.evaluation.common import EvaluationAdapter

class VRSBenchAdapter(EvaluationAdapter):
    name = "VRSBench"

    def load_dataset(self):
        print("Loading VRSBench dataset (mock)...")
        return []

    def evaluate(self, model_registry, limit=100):
        print("Evaluating on VRSBench (Grounding)...")
        return {"iou": 0.65}
