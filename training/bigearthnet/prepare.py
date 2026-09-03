import os
import json

def prepare_dataset(output_dir: str):
    """
    Simulates the preparation of the BigEarthNet dataset for instruction tuning.
    In reality, this would download the LMDB/GeoTIFFs and format them into
    JSONL files containing {"image": "...", "text": "..."}.
    """
    os.makedirs(output_dir, exist_ok=True)
    print("Preparing BigEarthNet instruction tuning dataset...")
    # Mock data generation
    data = [
        {"image": "patch_1.tif", "text": "This image contains continuous urban fabric and arable land."},
        {"image": "patch_2.tif", "text": "This image shows coniferous forest and a river."}
    ]
    with open(os.path.join(output_dir, "train.jsonl"), "w") as f:
        for item in data:
            f.write(json.dumps(item) + "\n")
    print("Dataset prepared.")

if __name__ == "__main__":
    prepare_dataset("data")
