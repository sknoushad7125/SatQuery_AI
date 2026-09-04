import subprocess
import sys
from pathlib import Path

def run_script(name):
    print(f"--- Running {name} ---")
    ret = subprocess.run([sys.executable, f"scripts/datasets/{name}"])
    if ret.returncode != 0:
        print(f"Error running {name}")
        sys.exit(1)

def main():
    Path("training_data/manifests").mkdir(parents=True, exist_ok=True)
    Path("training_data/reports").mkdir(parents=True, exist_ok=True)

    run_script("prepare_rsvqa.py")
    run_script("prepare_vrsbench.py")
    run_script("prepare_bigearthnet.py")
    run_script("prepare_cdvqa.py")

    print("--- Running validate_manifests.py ---")
    run_script("validate_manifests.py")

if __name__ == "__main__":
    main()
