import os
import json
from huggingface_hub import HfApi, HfFileSystem, get_hf_file_metadata, hf_hub_url
from datasets import get_dataset_config_names, load_dataset

def verify_access():
    print("Verifying HF Access...\n")
    api = HfApi()
    fs = HfFileSystem()

    # 1. BigEarthNet.txt
    print("--- 1. BIFOLD-BigEarthNetv2-0/BigEarthNet.txt ---")
    try:
        repo_id = "BIFOLD-BigEarthNetv2-0/BigEarthNet.txt"
        # Check configs
        configs = get_dataset_config_names(repo_id)
        print(f"Configs available: {configs}")
        
        # We need BigEarthNet.txt or the equivalent that has S1, S2, and text.
        # Let's see which config has it. If 'BigEarthNet.txt' is not a config, maybe 'all' or 'default'?
        config_to_use = "BigEarthNet.txt" if "BigEarthNet.txt" in configs else configs[0]
        print(f"Using config: {config_to_use}")
        
        ds = load_dataset(repo_id, name=config_to_use, streaming=True, trust_remote_code=True)
        print("Available splits:", list(ds.keys()))
        
        # Take the first train record
        first_split = list(ds.keys())[0]
        first_record = next(iter(ds[first_split]))
        
        print("\nSchema / Features:")
        features = ds[first_split].features
        for k, v in features.items():
            print(f"  {k}: {v}")
            
        print("\nFirst Record Summary:")
        for k, v in first_record.items():
            if hasattr(v, 'shape'):
                print(f"  {k}: shape={v.shape}, dtype={v.dtype}")
            elif isinstance(v, list):
                print(f"  {k}: list of length {len(v)} (e.g. {v[0] if v else ''})")
            elif isinstance(v, dict):
                print(f"  {k}: dict with keys {list(v.keys())}")
            else:
                val_str = str(v)[:100]
                print(f"  {k}: {val_str}")
                
        print("BigEarthNet Access: PASS\n")
    except Exception as e:
        print(f"BigEarthNet Access: FAIL ({e})\n")


    # 2. jirvin16/TEOChatlas
    print("--- 2. jirvin16/TEOChatlas ---")
    try:
        repo_id_teo = "jirvin16/TEOChatlas"
        files = api.list_repo_files(repo_id=repo_id_teo, repo_type="dataset")
        eval_files = [f for f in files if "eval" in f]
        print(f"Eval files found: {eval_files}")
        
        target_file = "eval/External_images.tar.gz"
        if target_file in eval_files:
            url = hf_hub_url(repo_id=repo_id_teo, filename=target_file, repo_type="dataset")
            meta = get_hf_file_metadata(url)
            size_mb = meta.size / (1024 * 1024)
            print(f"Target artifact '{target_file}' size: {size_mb:.2f} MB")
            
        print("TEOChatlas Access: PASS\n")
    except Exception as e:
        print(f"TEOChatlas Access: FAIL ({e})\n")

if __name__ == "__main__":
    verify_access()
