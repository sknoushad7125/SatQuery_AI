import os
import json
import hashlib

def verify_images():
    print("Verifying CDVQA Images...")
    
    with open("datasets/cdvqa/qa/Test_images.json") as f:
        target_ids = set(img["file_name"] for img in json.load(f)["images"])
        
    manifest_path = "datasets/cdvqa/images_manifest.json"
    if not os.path.exists(manifest_path):
        print("Manifest missing.")
        return False
        
    with open(manifest_path) as f:
        acquired = json.load(f)
        
    all_present = True
    for img_id in target_ids:
        if img_id not in acquired:
            all_present = False
            break
            
    if not all_present:
        print("[ERROR] Not all official CDVQA test image references exist locally.")
        return False
        
    for img_id, meta in acquired.items():
        path_a = os.path.join("datasets/cdvqa/images/A", img_id)
        path_b = os.path.join("datasets/cdvqa/images/B", img_id)
        
        if not os.path.exists(path_a) or not os.path.exists(path_b):
            print(f"[ERROR] Image {img_id} files missing.")
            return False
            
    print("[PASS] All acquired images verified successfully.")
    
    mapping_path = "datasets/cdvqa/cdvqa_mapping.json"
    if os.path.exists(mapping_path):
        with open(mapping_path) as f:
            mapping = json.load(f)
        
        if mapping.get("mapping_verified"):
            print("[PASS] Mapping is marked verified.")
        else:
            print("[WARNING] Mapping is not completely verified.")
    
    return True

if __name__ == "__main__":
    verify_images()
