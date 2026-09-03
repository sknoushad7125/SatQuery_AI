import os
import json
import hashlib
from datasets import load_dataset
import argparse

def compute_sha256(path):
    sha256_hash = hashlib.sha256()
    with open(path, "rb") as f:
        for byte_block in iter(lambda: f.read(4096), b""):
            sha256_hash.update(byte_block)
    return sha256_hash.hexdigest()

def acquire(verify_only=False):
    with open("datasets/cdvqa/qa/Test_images.json") as f:
        target_ids = set(img["file_name"] for img in json.load(f)["images"])
        
    print(f"Target unique test images: {len(target_ids)}")
    
    os.makedirs("datasets/cdvqa/images/A", exist_ok=True)
    os.makedirs("datasets/cdvqa/images/B", exist_ok=True)
    
    acquired = {}
    manifest_path = "datasets/cdvqa/images_manifest.json"
    if os.path.exists(manifest_path):
        with open(manifest_path) as f:
            acquired = json.load(f)
            
    if verify_only:
        print(f"Verifying {len(acquired)} downloaded images...")
        all_ok = True
        for k, v in acquired.items():
            path_a = os.path.join("datasets/cdvqa/images/A", k)
            path_b = os.path.join("datasets/cdvqa/images/B", k)
            if not os.path.exists(path_a) or not os.path.exists(path_b):
                print(f"Missing image {k}")
                all_ok = False
        if all_ok:
            print("All verified successfully.")
        return
        
    ds = load_dataset("ljx620/CDVQA", split="test", streaming=True)
    
    for row in ds:
        # the row contains json string
        meta_str = row.get("json", "{}")
        meta = meta_str.get("meta", {})
            
        img_id = meta.get("image_id")
        
        if img_id in target_ids and img_id not in acquired:
            from PIL import Image; import io; img_a = Image.open(io.BytesIO(row.get("0.img")))
            img_b = Image.open(io.BytesIO(row.get("1.img")))
            
            path_a = os.path.join("datasets/cdvqa/images/A", img_id)
            path_b = os.path.join("datasets/cdvqa/images/B", img_id)
            
            img_a.save(path_a)
            img_b.save(path_b)
            
            acquired[img_id] = {
                "cdvqa_file": img_id,
                "source": meta.get("source", "Unknown"),
                "width": img_a.width,
                "height": img_a.height,
                "channels": len(img_a.getbands()),
                "sha256_A": compute_sha256(path_a),
                "sha256_B": compute_sha256(path_b),
                "shard": row.get("__url__", "Unknown"),
                "metadata": meta
            }
            
            print(f"Acquired {len(acquired)}/{len(target_ids)}: {img_id}")
            
            # Save manifest incrementally
            with open(manifest_path, "w") as f:
                json.dump(acquired, f, indent=2)
                
            if len(acquired) >= len(target_ids):
                print("Acquired all required images!")
                break
                
    print("Download phase complete.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--verify", action="store_true")
    args = parser.parse_args()
    acquire(verify_only=args.verify)
