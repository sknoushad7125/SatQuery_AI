import os
import json
from abc import ABC, abstractmethod
from typing import Dict, Any, Union

class LLMProvider(ABC):
    @abstractmethod
    def generate(self, prompt: Union[str, list]) -> str:
        pass

class GeminiProvider(LLMProvider):
    def __init__(self):
        from google import genai
        api_key = os.environ.get("GEMINI_API_KEY")
        if not api_key:
            raise ValueError("GEMINI_API_KEY not set")
        self.client = genai.Client(api_key=api_key)
        self.model_name = 'gemini-1.5-flash'

    def generate(self, prompt: Union[str, list]) -> str:
        response = self.client.models.generate_content(
            model=self.model_name,
            contents=prompt
        )
        return response.text

class MockProvider(LLMProvider):
    def generate(self, prompt: Union[str, list]) -> str:
        text_prompt = prompt[0] if isinstance(prompt, list) and len(prompt) > 0 and isinstance(prompt[0], str) else prompt
        if not isinstance(text_prompt, str):
            text_prompt = str(text_prompt)

        if "You are the synthesis agent" in text_prompt or "Task Classification:" in text_prompt:
            if "Task Classification: vqa" in text_prompt:
                return "ANSWER\n- Based on the analysis, yes.\n\nOBSERVATIONS\n- Visual evidence confirms the answer.\n\nMODEL EVIDENCE\n- VQA\n- Yes\n- 0.99\n\nINTERPRETATION\n- The scene contains the requested elements."
            elif "Task Classification: captioning" in text_prompt:
                return "SCENE DESCRIPTION\n- An optical image.\n\nKEY OBSERVATIONS\n- Various elements detected in the scene.\n\nMODEL\n- BLIP RSICD"
            elif "Task Classification: change_detection" in text_prompt:
                return "CHANGE SUMMARY\n- Changes detected across the region.\n\nQUANTITATIVE RESULT\n- 15%\n- High confidence\n- 3 regions\n\nSPATIAL EVIDENCE\n- Crops show distinct structural changes.\n\nBEFORE -> AFTER INTERPRETATION\n- Buildings appeared in the empty lots."
            elif "Task Classification: optical_sar" in text_prompt:
                return "CROSS-MODAL ANALYSIS\n- Fusion of optical and SAR confirms land cover.\n\nOPTICAL EVIDENCE\n- Optical shows vegetation and built-up areas.\n\nSAR EVIDENCE\n- SAR backscatter confirms structural density.\n\nFUSION RESULT\n- Built-up\n\nUNCERTAINTY\n- High confidence, low uncertainty."
            elif "Task Classification: sar_classification" in text_prompt:
                return "CLASSIFICATION RESULT\n- Urban\n\nINTERPRETATION\n- The region is primarily characterized by the dominant classes."
            return "Synthesized answer based on the tool evidence."

        # Fallback for old routing
        return json.dumps({"task_type": "single_vqa", "reason": "Mock routing based on keywords."})

def get_llm_provider() -> LLMProvider:
    if os.environ.get("GEMINI_API_KEY"):
        return GeminiProvider()
    return MockProvider()
