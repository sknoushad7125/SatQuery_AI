import os

def acquire():
    print("HF_TOKEN is missing. Please configure it locally.")
    print("BigEarthNet.txt repository 'BIFOLD-BigEarthNetv2-0' is gated.")
    print("Run `huggingface-cli login` to authenticate.")

if __name__ == "__main__":
    acquire()
