import argparse
import os
import subprocess
import sys
import shutil

def check_credentials():
    creds_path = os.path.expanduser('~/.kaggle/kaggle.json')
    if not os.path.exists(creds_path):
        return False
    return True

def get_kaggle_cmd():
    # Use kaggle from the venv
    return os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), ".venv", "bin", "kaggle")

def verify():
    print("--- VERIFYING LEVIR-CD SOURCE ---")
    print("Expected dataset: mdrifaturrahman33/levir-cd (Original 637-pair Kaggle Mirror)")
    print("Authentication: Required (Kaggle API Token)")
    
    if not check_credentials():
        print("\n[BLOCKER] Kaggle credentials missing.")
        sys.exit(1)
        
    kaggle_bin = get_kaggle_cmd()
    try:
        res = subprocess.run([kaggle_bin, "datasets", "metadata", "mdrifaturrahman33/levir-cd"], capture_output=True, text=True)
        if res.returncode != 0:
            print(f"\n[BLOCKER] Kaggle API failed: {res.stderr}")
            sys.exit(1)
        print("\n[OK] Kaggle metadata verified.")
    except Exception as e:
        print(f"\n[BLOCKER] Unexpected error: {e}")
        sys.exit(1)

def download():
    print("--- DOWNLOADING LEVIR-CD ---")
    if not check_credentials():
        print("[BLOCKER] Kaggle credentials missing.")
        sys.exit(1)
        
    kaggle_bin = get_kaggle_cmd()
    os.makedirs("datasets/levir_cd", exist_ok=True)
    temp_dir = "datasets/levir_cd_temp"
    try:
        os.makedirs(temp_dir, exist_ok=True)
        print("Starting Kaggle download (2.46 GB)...")
        subprocess.run([kaggle_bin, "datasets", "download", "-d", "mdrifaturrahman33/levir-cd", "-p", temp_dir, "--unzip"], check=True)
        
        # The Kaggle structure is usually: LEVIR CD/test/...
        # Also could just be test/A/... if the folder collapsed
        # Let's find the root that contains train/val/test
        found_base = None
        for root, dirs, files in os.walk(temp_dir):
            if "train" in dirs and "val" in dirs and "test" in dirs:
                found_base = root
                break
                
        if not found_base:
            print("[BLOCKER] Downloaded archive does not contain train/val/test directories.")
            sys.exit(1)
            
        print(f"Extracting splits from {found_base} to datasets/levir_cd/ ...")
        for split in ["train", "val", "test"]:
            for mod in ["A", "B", "label"]:
                src = os.path.join(found_base, split, mod)
                dst = os.path.join("datasets", "levir_cd", split, mod)
                if os.path.exists(src):
                    os.makedirs(os.path.dirname(dst), exist_ok=True)
                    if os.path.exists(dst): shutil.rmtree(dst)
                    shutil.move(src, dst)
                    
        shutil.rmtree(temp_dir)
        print("[OK] Download and structural extraction complete.")
    except Exception as e:
        print(f"[BLOCKER] Download failed: {e}")
        sys.exit(1)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--verify", action="store_true")
    parser.add_argument("--download", action="store_true")
    args = parser.parse_args()
    
    if args.verify:
        verify()
    if args.download:
        download()
