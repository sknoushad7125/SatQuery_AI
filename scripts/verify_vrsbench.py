import os
import json

def verify():
    anno_path = "datasets/vrsbench/qa/VRSBench_EVAL_referring.json"
    if not os.path.exists(anno_path):
        print("[ERROR] Annotation file missing.")
        return
        
    with open(anno_path) as f:
        annos = json.load(f)
        
    print("VRSBench Verification:")
    print(f"Total annotations parsed: {len(annos)}")
    
    unique_imgs = set(a["image_id"] for a in annos)
    print(f"Total unique images referenced: {len(unique_imgs)}")
    
    manifest_path = "datasets/vrsbench/manifest.json"
    if os.path.exists(manifest_path):
        with open(manifest_path) as f:
            manifest = json.load(f)
            
        downloaded = 0
        missing = 0
        
        for img in unique_imgs:
            if img in manifest["records"]:
                img_path = os.path.join("datasets/vrsbench/images", img)
                if os.path.exists(img_path):
                    downloaded += 1
                else:
                    missing += 1
            else:
                missing += 1
                
        print(f"Images downloaded and verified on disk: {downloaded}")
        print(f"Images missing: {missing}")
    else:
        print("[INFO] No manifest found. Image download was deferred.")
    
    if len(annos) > 0:
        first = annos[0]
        print("\nSchema Check (First Annotation):")
        print(f"Image ID format: {first.get('image_id')}")
        print(f"Ground Truth (String bin format): {first.get('ground_truth')}")
        print(f"Corner Bbox (Float): {first.get('obj_corner')}")
        print(f"Object Class: {first.get('obj_cls')}")
        
if __name__ == "__main__":
    verify()
