import os
import sys
import json
import traceback

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from backend.tools.vqa.real_vqa import RealVQATool
from src.tools.single_image_vqa import SingleImageVQATool
from src.agent.schemas import ToolRequest, ImageInput

def run_validation():
    print("Loading RSVQA Validation Data...")
    
    q_path = "datasets/rsvqa/LR_split_val_questions.json"
    a_path = "datasets/rsvqa/LR_split_val_answers.json"
    img_dir = "datasets/rsvqa/Images_LR"
    
    with open(q_path) as f:
        questions = json.load(f)["questions"]
        
    with open(a_path) as f:
        answers = json.load(f)["answers"]
        
    # Map answers by question_id
    ans_map = {}
    for a in answers:
        if "question_id" in a and "answer" in a:
            ans_map[a["question_id"]] = a["answer"]
    
    # Init both tools
    print("Initializing Phase 4 Source Tool...")
    source_tool = SingleImageVQATool(checkpoint_path="checkpoints/vqa_baseline.pth")
    
    print("Initializing Backend Adapter Tool...")
    backend_tool = RealVQATool()
    
    print("\n--- Running 10 Samples ---")
    
    correct_count = 0
    total_tested = 0
    mismatches_between_tools = 0
    
    for q in questions:
        if total_tested >= 10:
            break
            
        img_id = q.get("img_id")
        img_path = os.path.join(img_dir, f"{img_id}.tif")
        
        if not os.path.exists(img_path):
            continue
            
        gt_answer = ans_map.get(q["id"])
        if not gt_answer:
            continue
            
        question_text = q["question"]
        
        # Run Phase 4 Source
        req = ToolRequest(tool_name="single_image_vqa", query=question_text, images=[ImageInput(filepath=img_path, modality="optical")])
        source_result = source_tool.execute(req)
        
        if not source_result.success:
            print(f"Source Tool Failed on {img_path}: {source_result.error}")
            continue
            
        source_pred = source_result.data.get("answer")
        
        # Run Backend Adapter
        class MockMeta:
            modality = "optical"
        
        backend_result = backend_tool.run([img_path], question_text)
        
        if backend_result.status != "success":
            print(f"Backend Tool Failed on {img_path}: {backend_result.error_message}")
            continue
            
        backend_pred = backend_result.text
        conf = backend_result.confidence
        
        exact_match = (backend_pred == gt_answer)
        tool_match = (backend_pred == source_pred)
        
        print(f"\nSample {total_tested + 1}: Image {img_id}")
        print(f"Q: {question_text}")
        print(f"GT Answer: {gt_answer}")
        print(f"Predicted Answer (Backend): {backend_pred} (Conf: {conf:.4f})")
        print(f"Predicted Answer (Source):  {source_pred}")
        print(f"Ground Truth Exact Match:   {exact_match}")
        print(f"Adapter Matches Source:     {tool_match}")
        
        if exact_match:
            correct_count += 1
        if not tool_match:
            mismatches_between_tools += 1
            
        total_tested += 1

    print("\n=== SUMMARY ===")
    print(f"Total Tested: {total_tested}")
    print(f"Accuracy: {correct_count}/{total_tested} ({(correct_count/total_tested)*100:.2f}%)")
    print(f"Adapter = Source Mismatches: {mismatches_between_tools}")
    
    if mismatches_between_tools > 0:
        print("FAIL: Backend adapter output differs from original Phase 4 tool!")
        sys.exit(1)
    else:
        print("PASS: Backend adapter exactly replicates Phase 4 inference!")

if __name__ == "__main__":
    run_validation()
