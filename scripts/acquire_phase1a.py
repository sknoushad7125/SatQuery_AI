import urllib.request
import json
import os
import hashlib
import zipfile
from datetime import datetime, timezone
import ssl

ctx = ssl.create_default_context()
ctx.check_hostname = False
ctx.verify_mode = ssl.CERT_NONE

def compute_sha256(path):
    if not os.path.exists(path): return None
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()

def download_file(url, dest):
    print(f"  -> Downloading: {url}")
    try:
        req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0 (SatQuery AI Phase 1A)'})
        with urllib.request.urlopen(req, context=ctx, timeout=30) as response:
            with open(dest, 'wb') as f:
                for chunk in iter(lambda: response.read(8192), b""):
                    f.write(chunk)
        return True, None
    except urllib.error.HTTPError as e:
        return False, f"HTTP Error {e.code}"
    except Exception as e:
        return False, str(e)

def acquire_cdvqa():
    print("\n--- ACQUIRING CDVQA ---")
    base_url = "https://raw.githubusercontent.com/YZHJessica/CDVQA/main"
    files = ["Test_questions.json", "Test_answers.json", "Test_images.json"]
    out_dir = "datasets/cdvqa/qa"
    os.makedirs(out_dir, exist_ok=True)
    
    manifest = {
        "dataset": "CDVQA",
        "source": "https://github.com/YZHJessica/CDVQA",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "files": [],
        "metrics": {}
    }
    
    for f in files:
        dest = os.path.join(out_dir, f)
        if not os.path.exists(dest):
            ok, err = download_file(f"{base_url}/{f}", dest)
            if not ok:
                print(f"  [ERROR] Failed to download {f}: {err}")
                return
                
        manifest["files"].append({
            "name": f,
            "sha256": compute_sha256(dest)
        })
    
    with open(os.path.join(out_dir, "Test_questions.json")) as f: qs = json.load(f)["questions"]
    with open(os.path.join(out_dir, "Test_answers.json")) as f: ans = json.load(f)["answers"]
    with open(os.path.join(out_dir, "Test_images.json")) as f: imgs = json.load(f)["images"]
    
    sample_img = imgs[0]['file_name']
    is_levir_format = sample_img.endswith(".png")
    
    manifest["metrics"] = {
        "questions_count": len(qs),
        "answers_count": len(ans),
        "images_referenced": len(imgs),
        "sample_image_id": sample_img,
        "matches_levir_format": is_levir_format,
        "question_fields": list(qs[0].keys()),
        "answer_fields": list(ans[0].keys())
    }
    
    with open("datasets/cdvqa/manifest.json", "w") as f:
        json.dump(manifest, f, indent=2)
    print("  [OK] CDVQA Verification & Manifest Complete")

def acquire_rsvqa():
    print("\n--- ACQUIRING RSVQA-LR ---")
    base_url = "https://zenodo.org/records/6344333/files"
    files = ["Images_LR.zip", "Questions_LR.zip", "Answers_LR.zip"]
    out_dir = "datasets/rsvqa"
    os.makedirs(f"{out_dir}/images", exist_ok=True)
    os.makedirs(f"{out_dir}/qa", exist_ok=True)
    
    manifest = {
        "dataset": "RSVQA-LR",
        "source": "https://doi.org/10.5281/zenodo.6344333",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "files": [],
        "splits": {},
        "blockers": None
    }
    
    for f in files:
        dest = os.path.join(out_dir, f)
        if not os.path.exists(dest):
            ok, err = download_file(f"{base_url}/{f}", dest)
            if not ok:
                msg = f"Zenodo Download Failed ({err}). Manual browser download required from {base_url}/{f}"
                print(f"  [ERROR] {msg}")
                manifest["blockers"] = msg
                with open(os.path.join(out_dir, "manifest.json"), "w") as mf:
                    json.dump(manifest, mf, indent=2)
                return
                
        manifest["files"].append({
            "name": f,
            "sha256": compute_sha256(dest)
        })

    print("  -> Extracting RSVQA-LR...")
    
    if os.path.exists(f"{out_dir}/Questions_LR.zip"):
        with zipfile.ZipFile(f"{out_dir}/Questions_LR.zip", 'r') as z:
            z.extractall(f"{out_dir}/qa")
            for fname in z.namelist():
                if fname.endswith(".json") and "split" in fname:
                    with open(os.path.join(out_dir, "qa", fname)) as jf:
                        data = json.load(jf)
                        manifest["splits"][fname] = len(data.get("questions", data))
    
    if os.path.exists(f"{out_dir}/Images_LR.zip"):
        with zipfile.ZipFile(f"{out_dir}/Images_LR.zip", 'r') as z:
            z.extractall(f"{out_dir}/images")
            img_count = sum(1 for f in z.namelist() if f.endswith(('.png', '.tif')))
            manifest["total_images_extracted"] = img_count

    with open(os.path.join(out_dir, "manifest.json"), "w") as f:
        json.dump(manifest, f, indent=2)
    print("  [OK] RSVQA-LR Verification & Manifest Complete")

def get_dir_size(path):
    total = 0
    for dirpath, dirnames, filenames in os.walk(path):
        for f in filenames:
            fp = os.path.join(dirpath, f)
            if not os.path.islink(fp):
                total += os.path.getsize(fp)
    return total

if __name__ == "__main__":
    acquire_cdvqa()
    acquire_rsvqa()
    
    print("\n--- DISK USAGE ---")
    print(f"CDVQA: {get_dir_size('datasets/cdvqa') / 1024 / 1024:.2f} MB")
    print(f"RSVQA: {get_dir_size('datasets/rsvqa') / 1024 / 1024:.2f} MB")
