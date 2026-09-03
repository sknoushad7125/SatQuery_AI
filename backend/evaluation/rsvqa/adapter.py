from backend.evaluation.common import EvaluationAdapter

class RSVQAAdapter(EvaluationAdapter):
    name = "RSVQA"

    def load_dataset(self):
        print("Loading RSVQA dataset (mock)...")
        return []

    def evaluate(self, model_registry, limit=100):
        print("Evaluating on RSVQA...")
        # Mock evaluation
        return {"accuracy": 0.82}
