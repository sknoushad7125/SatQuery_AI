import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from backend.tools.optical_sar.real_fusion import RealDecisionFusionTool
from backend.api.schemas.domain import ImageMetadata

def run_test():
    tool = RealDecisionFusionTool()
    
    # Needs two images. Let's use test_1.png from levir_cd for both just to see if it runs
    img1 = "datasets/levir_cd/test/A/test_1.png"
    img2 = "datasets/levir_cd/test/B/test_1.png"
    
    res1 = tool.run([img1, img2], "analyze fusion")
    print("Result 1:", res1.text)
    print("Probs 1:", res1.structured_data["fused_class_probabilities"])
    print("Status:", res1.metadata)
    
if __name__ == "__main__":
    run_test()
