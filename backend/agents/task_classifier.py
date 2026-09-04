import json
from typing import List
from backend.api.schemas.domain import AnalysisInput, AnalysisQuery, AnalysisTask
from backend.services.llm_provider import get_llm_provider

class TaskClassifier:
    def __init__(self):
        self.llm = get_llm_provider()

    def classify(self, query: AnalysisQuery, input_data: AnalysisInput) -> AnalysisTask:
        """
        Classifies the query and input into a specific task type using LLM.
        """
        # If the input implicitly forces a task, we could override, but let's let LLM decide.
        prompt = f"""
You are the routing agent for SatQuery AI.
Input type provided by user: {input_data.input_type} (options: single, temporal_pair, optical_sar_pair)
User Query: {query.text}

Available Tasks:
- single_vqa: Answer general questions about a single image.
- captioning: Describe a single image.
- grounding: Highlight or locate an object in a single image.
- bi_temporal_change: Detect changes between two temporal images.
- change_vqa: Answer questions about changes between two temporal images.
- optical_sar_analysis: Analyze a region using both optical and SAR images.

Respond with ONLY a JSON object containing "task_type" and "reason".
"""
        try:
            response = self.llm.generate(prompt)
            # Basic cleanup if markdown backticks are returned
            if response.startswith("```json"):
                response = response[7:-3]
            elif response.startswith("```"):
                response = response[3:-3]

            data = json.loads(response.strip())
            task_type = data.get("task_type", "unsupported")

            # Validation safeguard
            valid_tasks = ["single_vqa", "captioning", "grounding", "bi_temporal_change", "change_vqa", "optical_sar_analysis", "multi_tool"]
            if task_type not in valid_tasks:
                task_type = "unsupported"

            return AnalysisTask(task_type=task_type)
        except Exception as e:
            print(f"LLM Classification failed: {e}. Falling back to default.")
            return AnalysisTask(task_type="unsupported")
