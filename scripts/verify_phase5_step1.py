import os
import sys
import traceback

# Add project root to path for backend imports
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from backend.tools.change.real_change import RealSemanticChangeTool

class MockImageMetadata:
    def __init__(self, modality):
        self.modality = modality

def run_test():
    tool = RealSemanticChangeTool()
    
    img_a = "datasets/levir_cd/test/A/test_1.png"
    img_b = "datasets/levir_cd/test/B/test_1.png"
    
    if not os.path.exists(img_a) or not os.path.exists(img_b):
        print("Test images not found!")
        return
        
    print(f"Testing with:\n - {img_a}\n - {img_b}")
    
    meta_a = MockImageMetadata(modality="optical")
    meta_b = MockImageMetadata(modality="optical")
    
    is_valid = tool.validate([meta_a, meta_b], "what changed?")
    print(f"Validation passed: {is_valid}")
    if not is_valid:
        print("Validation failed.")
        return
        
    print("Executing tool...")
    try:
        result = tool.run([img_a, img_b], "what changed?")
    except Exception as e:
        print(f"Tool run raised an exception!")
        traceback.print_exc()
        return

    print(f"\n--- RESULTS ---")
    print(f"Status: {result.status}")
    
    if result.status == "success":
        print(f"Model Name: {result.model_name}")
        print(f"Text Output: {result.text}")
        print(f"Confidence: {result.confidence:.4f}")
        perc = result.evidence.get("percentage_change", 0)
        print(f"Percentage Change: {perc:.4f}%")
        print(f"Checkpoint Used: {result.metadata.get('checkpoint')}")
        
        # Verify conditions
        assert result.model_name == "SiameseResNet18 Phase4", "Wrong model executed!"
        assert "best_bce_dice.pth" in result.metadata.get("checkpoint", ""), "Wrong checkpoint!"
        assert perc >= 0.0, "Invalid percentage"
        print("\nSUCCESS: All conditions met. The verified SiameseResNet18 change detector is successfully connected to the backend API tool!")
    else:
        # We don't use result.error_message because Pydantic might not allow it as an attribute
        dump = result.model_dump()
        print(f"Error output from tool: {dump}")

if __name__ == "__main__":
    run_test()
