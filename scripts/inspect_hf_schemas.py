import sys
import os
sys.path = [p for p in sys.path if not p.endswith('/app') and p != '']

from datasets import load_dataset

def inspect_ds(name, split="train"):
    try:
        ds = load_dataset(name, split=split, streaming=True)
        sample = next(iter(ds))
        print(f"--- {name} ---")
        for k, v in sample.items():
            print(f"  {k}: {type(v)}")
    except Exception as e:
        print(f"Failed {name}: {e}")

inspect_ds("BIFOLD-BigEarthNetv2-0/BigEarthNet.txt")
inspect_ds("ericyu/LEVIRCD_Cropped_256")
inspect_ds("saaketht/rsvqa_lq")
inspect_ds("mespinosami/sen12mscr")
inspect_ds("xiang709/VRSBench")
