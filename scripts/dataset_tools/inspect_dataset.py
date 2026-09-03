import sys
import os
from validate_levir import validate_levir
from validate_sen12ms import validate_sen12ms
from validate_rsvqa import validate_rsvqa
from validate_bigearthnet import validate_bigearthnet

def main():
    datasets = [
        {"name": "LEVIR-CD", "func": validate_levir, "path": "datasets/levir_cd"},
        {"name": "SEN12MS", "func": validate_sen12ms, "path": "datasets/sen12ms"},
        {"name": "RSVQA", "func": validate_rsvqa, "path": "datasets/rsvqa"},
        {"name": "BigEarthNet", "func": validate_bigearthnet, "path": "datasets/bigearthnet"},
        # VRSBench and CDVQA are derived/zero-shot proxies without custom validators for now
        {"name": "VRSBench", "func": lambda p: {"status": "ABSENT", "samples": "-", "valid": False}, "path": "datasets/vrsbench"},
        {"name": "CDVQA", "func": lambda p: {"status": "ABSENT", "samples": "-", "valid": False}, "path": "datasets/cdvqa"}
    ]
    
    print(f"{'Dataset':<15} {'Status':<10} {'Samples':<10} {'Valid':<10} {'Location':<25}")
    print("-" * 75)
    
    for ds in datasets:
        try:
            res = ds["func"](ds["path"]) if "func" in ds else ds["func"](ds["path"])
        except Exception:
            res = ds["func"](ds["path"])
            
        status = res.get("status", "ERROR")
        samples = str(res.get("samples", "-"))
        valid = str(res.get("valid", "-"))
        
        print(f"{ds['name']:<15} {status:<10} {samples:<10} {valid:<10} {ds['path']:<25}")

if __name__ == "__main__":
    main()
