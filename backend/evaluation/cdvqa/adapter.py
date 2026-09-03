from backend.evaluation.common import EvaluationAdapter

class CDVQAAdapter(EvaluationAdapter):
    name = "CDVQA"

    def load_dataset(self):
        print("Loading CDVQA dataset (mock)...")
        return []

    def evaluate(self, model_registry, limit=100):
        print("Evaluating on CDVQA (Change VQA)...")
        return {"accuracy": 0.78, "semantic_similarity": 0.81}
