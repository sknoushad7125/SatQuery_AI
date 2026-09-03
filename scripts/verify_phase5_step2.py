import os
import sys
import traceback

# Add project root to path for backend imports
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from backend.tools.vqa.real_vqa import RealVQATool

class MockImageMetadata:
    def __init__(self, modality):
        self.modality = modality

def run_test():
    tool = RealVQATool()
    
    img_path = "datasets/levir_cd/test/A/test_1.png"
    
    if not os.path.exists(img_path):
        print(f"Test image not found at {img_path}!")
        return
        
    print(f"Testing with image: {img_path}")
    
    meta = MockImageMetadata(modality="optical")
    
    questions = [
        "are there buildings ?",
        "is there a road ?",
        "what type of scene is this ?"
    ]
    
    for q in questions:
        print(f"\n--- Question: '{q}' ---")
        
        is_valid = tool.validate([meta], q)
        if not is_valid:
            print("Validation failed.")
            continue
            
        try:
            result = tool.run([img_path], q)
        except Exception as e:
            print("Tool run raised an exception!")
            traceback.print_exc()
            continue

        print(f"Status: {result.status}")
        
        if result.status == "success":
            print(f"Model Name: {result.model_name}")
            print(f"Text Output (Answer): {result.text}")
            print(f"Confidence: {result.confidence:.4f}")
            print(f"Checkpoint Used: {result.metadata.get('checkpoint')}")
            
            # Verify conditions
            assert result.model_name == "Custom-RSVQA-ResNet18-GRU", "Wrong model executed!"
            assert "vqa_baseline.pth" in result.metadata.get("checkpoint", ""), "Wrong checkpoint!"
        else:
            dump = result.model_dump()
            print(f"Error output from tool: {dump}")

if __name__ == "__main__":
    run_test()
