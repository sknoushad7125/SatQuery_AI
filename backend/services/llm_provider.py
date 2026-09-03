import os
import json
from abc import ABC, abstractmethod
from typing import Dict, Any

class LLMProvider(ABC):
    @abstractmethod
    def generate(self, prompt: str) -> str:
        pass

class GeminiProvider(LLMProvider):
    def __init__(self):
        import google.generativeai as genai
        api_key = os.environ.get("GEMINI_API_KEY")
        if not api_key:
            raise ValueError("GEMINI_API_KEY not set")
        genai.configure(api_key=api_key)
        self.model = genai.GenerativeModel('gemini-1.5-pro')

    def generate(self, prompt: str) -> str:
        response = self.model.generate_content(prompt)
        return response.text

class MockProvider(LLMProvider):
    def generate(self, prompt: str) -> str:
        # Simple heuristic to mock the LLM routing
        # Extract just the user query from the prompt
        query_line = [line for line in prompt.split('\n') if line.startswith("User Query:")]
        user_query = query_line[0].lower() if query_line else prompt.lower()
        
        if "highlight" in user_query or "where" in user_query:
            task = "grounding"
        elif "describe" in user_query:
            task = "captioning"
        elif "change" in user_query or "increase" in user_query:
            if "?" in user_query:
                task = "change_vqa"
            else:
                task = "bi_temporal_change"
        elif "optical" in user_query and "sar" in user_query:
            task = "optical_sar_analysis"
        else:
            task = "single_vqa"
            
        return json.dumps({"task_type": task, "reason": "Mock routing based on keywords."})

def get_llm_provider() -> LLMProvider:
    if os.environ.get("GEMINI_API_KEY"):
        return GeminiProvider()
    return MockProvider()
