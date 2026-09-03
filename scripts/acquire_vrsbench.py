import os
import json
import zipfile
import hashlib
from huggingface_hub import hf_hub_download

def compute_sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()

def acquire_vrsbench():
    print("Starting VRSBench Acquisition...")
    
    anno_path = "datasets/vrsbench/qa/VRSBench_EVAL_referring.json"
    with open(anno_path) as f:
        annos = json.load(f)
        
    target_ids = list(set(img["image_id"] for img in annos))
    print(f"Required unique images for EVAL: {len(target_ids)}")
    
    print("Downloading Images_val.zip...")
    archive_path = hf_hub_download(
        repo_id="xiang709/VRSBench",
        repo_type="dataset",
        filename="Images_val.zip"
    )
    
    print(f"Archive downloaded/cached at: {archive_path}")
    
    out_dir = "datasets/vrsbench/images"
    os.makedirs(out_dir, exist_ok=True)
    
    manifest_path = "datasets/vrsbench/images_manifest.json"
    manifest = {"dataset": "VRSBench", "records": {}}
    
    extracted_count = 0
    missing_count = 0
    
    print("Extracting required VRSBench images...")
    with zipfile.ZipFile(archive_path, "r") as z:
        members = set(z.namelist())
        for img_id in target_ids:
            zip_path = f"Images_val/{img_id}"
            out_path = os.path.join(out_dir, img_id)
            
            if zip_path in members:
                # Extract if not exists
                if not os.path.exists(out_path):
                    with open(out_path, "wb") as f_out:
                        f_out.write(z.read(zip_path))
                
                manifest["records"][img_id] = {
                    "source": "xiang709/VRSBench/Images_val.zip",
                    "sha256": compute_sha256(out_path),
                    "file": img_id
                }
                extracted_count += 1
            else:
                print(f"[WARNING] {img_id} not found in archive!")
                missing_count += 1
                
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)
        
    print(f"Extracted {extracted_count} images. Missing: {missing_count}.")
    
if __name__ == "__main__":
    acquire_vrsbench()
