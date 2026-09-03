import json
import os
import evaluate as hf_evaluate
from abc import ABC, abstractmethod

class EvaluationAdapter(ABC):
    @abstractmethod
    def evaluate(self, model_function, dataset_path: str):
        pass

class RSVQAAdapter(EvaluationAdapter):
    def evaluate(self, model_function, dataset_path: str):
        if not os.path.exists(dataset_path):
            return {"status": "Not Evaluated", "reason": "Dataset not found"}
        
        try:
            with open(os.path.join(dataset_path, "questions.json")) as f:
                questions = json.load(f)
                
            em_metric = hf_evaluate.load("exact_match")
            preds = []
            refs = []
            
            for q in questions[:50]: # Limit for demo
                img_path = os.path.join(dataset_path, "images", f"{q['img_id']}.tif")
                if not os.path.exists(img_path): continue
                
                # Execute model
                result = model_function([img_path], q["question"])
                preds.append(result.text.strip().lower())
                refs.append(q["answer"].strip().lower())
                
            score = em_metric.compute(predictions=preds, references=refs)
            return {"status": "Evaluated", "accuracy": score["exact_match"]}
        except Exception as e:
            return {"status": "Error", "reason": str(e)}

class VRSBenchAdapter(EvaluationAdapter):
    def evaluate(self, model_function, dataset_path: str):
        if not os.path.exists(dataset_path):
            return {"status": "Not Evaluated", "reason": "Dataset not found"}
        return {"status": "Error", "reason": "IoU metric logic requires bounding box intersections not implemented in this snippet."}

class CDVQAAdapter(EvaluationAdapter):
    def evaluate(self, model_function, dataset_path: str):
        if not os.path.exists(dataset_path):
            return {"status": "Not Evaluated", "reason": "Dataset not found"}
        return {"status": "Error", "reason": "CDVQA data format handling not fully implemented."}

def run_evaluations():
    adapters = {
        "RSVQA": RSVQAAdapter(),
        "VRSBench": VRSBenchAdapter(),
        "CDVQA": CDVQAAdapter()
    }
    
    results = {}
    from backend.tools.vqa.real_vqa import RealVQATool
    vqa_tool = RealVQATool()
    
    def model_func(imgs, q):
        return vqa_tool.run(imgs, q)
        
    for name, adapter in adapters.items():
        res = adapter.evaluate(model_func, f"/app/datasets/{name.lower()}")
        results[name] = res
        
    print("\n--- Evaluation Results ---")
    for name, res in results.items():
        print(f"{name}: {res}")
        
if __name__ == "__main__":
    run_evaluations()
