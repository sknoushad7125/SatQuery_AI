import os
import json
import tarfile
import hashlib
import shutil
from huggingface_hub import hf_hub_download

def compute_sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()

def acquire_cdvqa():
    print("Starting CDVQA Acquisition (TEOChatlas)...")
    
    with open("datasets/cdvqa/qa/Test_images.json") as f:
        target_ids = set(img["file_name"] for img in json.load(f)["images"])
        
    print(f"Required unique images: {len(target_ids)}")
    
    print("Downloading External_images.tar.gz (this may take a while)...")
    archive_path = hf_hub_download(
        repo_id="jirvin16/TEOChatlas",
        repo_type="dataset",
        filename="eval/External_images.tar.gz"
    )
    
    print(f"Archive downloaded/cached at: {archive_path}")
    print(f"Archive size: {os.path.getsize(archive_path) / (1024*1024):.2f} MB")
    
    out_dir = "datasets/cdvqa/images"
    os.makedirs(out_dir, exist_ok=True)
    
    manifest_path = "datasets/cdvqa/images_manifest.json"
    manifest = {"dataset": "CDVQA_TEOChatlas", "records": {}}
    
    extracted_count = 0
    
    print("Extracting required CDVQA images...")
    with tarfile.open(archive_path, "r:gz") as tar:
        for member in tar:
            if "CDVQA" in member.name and member.name.endswith(".png"):
                basename = os.path.basename(member.name)
                # Check if it belongs to our target IDs
                # Note: The files might be named `07308_A.png` or similar, we check prefix
                prefix = basename.split("_")[0] + ".png" if "_" in basename else basename
                if prefix in target_ids or basename in target_ids:
                    # Extract to a temp flat file
                    member.name = basename
                    tar.extract(member, out_dir)
                    
                    out_path = os.path.join(out_dir, basename)
                    
                    manifest["records"][basename] = {
                        "source": "jirvin16/TEOChatlas",
                        "sha256": compute_sha256(out_path),
                        "file": basename
                    }
                    extracted_count += 1
                
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)
        
    print(f"Extracted {extracted_count} physical image files.")
    
    # We don't remove the archive here because hf_hub_download manages it in ~/.cache/huggingface.
    # The cache should be managed by the user or HF tools, but we can instruct the user on how to clear it.
    
if __name__ == "__main__":
    acquire_cdvqa()
