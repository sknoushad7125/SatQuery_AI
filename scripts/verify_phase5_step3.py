import os
import sys
import json
import traceback
from unittest.mock import patch, MagicMock

# Mock peft so that backend registry doesn't crash on import
sys.modules['peft'] = MagicMock()

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from backend.agents.controller import AgentController
from backend.api.schemas.domain import AnalysisQuery, AnalysisInput, ImageMetadata, ToolResult
from backend.services.llm_provider import LLMProvider

class MockSynthesisProvider(LLMProvider):
    def __init__(self, should_fail=False):
        self.should_fail = should_fail
        self.last_prompt = None
        self.synthesis_called = False
        
    def generate(self, prompt: str) -> str:
        self.last_prompt = prompt
        
        # If it's a classification prompt
        if "You are the routing agent" in prompt:
            return json.dumps({"task_type": "single_vqa", "reason": "mock classification"})
            
        # If it's the synthesis prompt
        if "You are the synthesis agent" in prompt:
            self.synthesis_called = True
            if self.should_fail:
                raise Exception("Simulated Gemini API Failure")
            return "SYNTHESIS_TEST_RESPONSE"
            
        return ""

class MockPlannerSuccess:
    def plan(self, task):
        class MockTool:
            name = "MockChangeTool"
            def validate(self, images, query): return True
            def run(self, images, query):
                return ToolResult(
                    tool_name="MockChangeTool",
                    model_name="best_bce_dice.pth",
                    model_type="real",
                    status="success",
                    text="Detected change",
                    confidence=0.91,
                    evidence={"percentage_change": 5.4, "huge_array": list(range(100))},
                    metadata={"checkpoint": "best_bce_dice.pth"}
                )
        return [MockTool()]

class MockPlannerFailure:
    def plan(self, task):
        class MockTool:
            name = "MockChangeTool"
            def validate(self, images, query): return True
            def run(self, images, query):
                return ToolResult(
                    tool_name="MockChangeTool",
                    model_name="best_bce_dice.pth",
                    model_type="real",
                    status="failure",
                    error_message="Model checkpoint missing"
                )
        return [MockTool()]

def run_test():
    img_path = "datasets/levir_cd/test/A/test_1.png"
    if not os.path.exists(img_path):
        print("Test image missing!")
        return

    query = AnalysisQuery(text="is there a road ?")
    img_meta = ImageMetadata(
        filename="test_1.png",
        format="PNG",
        width=1024,
        height=1024,
        bands=3,
        modality="optical"
    )
    input_data = AnalysisInput(input_type="single", images=[img_meta])
    
    # --- TEST 1: SUCCESS PATH ---
    print("\n--- TEST 1: SYNTHESIS SUCCESS ---")
    mock_success = MockSynthesisProvider(should_fail=False)
    with patch("backend.agents.task_classifier.get_llm_provider", return_value=mock_success):
        with patch("backend.agents.controller.get_llm_provider", return_value=mock_success):
            controller = AgentController()
            controller.planner = MockPlannerSuccess()
            trace = controller.process(query, input_data, [img_path])
            
            print(f"Final Result: {trace.final_result}")
            assert trace.final_result == "SYNTHESIS_TEST_RESPONSE", "Synthesis didn't return expected response"
            
            p = mock_success.last_prompt
            assert p is not None
            assert "5.4" in p, "percentage_change (5.4) missing from prompt!"
            assert "best_bce_dice.pth" in p, "metadata missing from prompt!"
            assert "success" in p, "status missing from prompt!"
            assert "0.91" in p, "confidence missing from prompt!"
            assert "<List truncated" in p, "large arrays should be truncated!"
            print("TEST 1 PASSED!")

    # --- TEST 2: FAILED TOOL ---
    print("\n--- TEST 2: FAILED TOOL INJECTION ---")
    mock_fail_tool = MockSynthesisProvider(should_fail=False)
    with patch("backend.agents.task_classifier.get_llm_provider", return_value=mock_fail_tool):
        with patch("backend.agents.controller.get_llm_provider", return_value=mock_fail_tool):
            controller2 = AgentController()
            controller2.planner = MockPlannerFailure()
            trace2 = controller2.process(query, input_data, [img_path])
            
            p = mock_fail_tool.last_prompt
            assert p is not None
            assert "failure" in p, "status='failure' missing from prompt!"
            assert "Model checkpoint missing" in p, "error_message missing from prompt!"
            print("TEST 2 PASSED!")

if __name__ == "__main__":
    run_test()
