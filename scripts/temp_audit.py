import urllib.request
import tarfile
import json
import ssl
import sys

ctx = ssl.create_default_context()
ctx.check_hostname = False
ctx.verify_mode = ssl.CERT_NONE

def audit_tar(url, max_files=100):
    print(f"Streaming {url}...", file=sys.stderr)
    req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
    sources = set()
    sizes = set()
    
    with urllib.request.urlopen(req, context=ctx) as response:
        with tarfile.open(fileobj=response, mode="r|") as tar:
            count = 0
            for member in tar:
                if member.name.endswith('.json'):
                    f = tar.extractfile(member)
                    if f:
                        data = json.loads(f.read().decode('utf-8'))
                        meta = data.get("meta", {})
                        src = meta.get("source", "Unknown")
                        wh = str(meta.get("wh", "Unknown"))
                        sources.add(src)
                        sizes.add(f"{src}: {wh}")
                        
                        count += 1
                        if count >= max_files:
                            break
    return sources, sizes

if __name__ == "__main__":
    sources = set()
    sizes = set()
    urls = [
        "https://huggingface.co/datasets/ljx620/CDVQA/resolve/main/test/test-00000.tar",
        "https://huggingface.co/datasets/ljx620/CDVQA/resolve/main/train/train-00000.tar",
        "https://huggingface.co/datasets/ljx620/CDVQA/resolve/main/val/val-00000.tar",
        "https://huggingface.co/datasets/ljx620/CDVQA/resolve/main/val/val-00010.tar"
    ]
    for url in urls:
        try:
            s1, s2 = audit_tar(url)
            sources.update(s1)
            sizes.update(s2)
        except Exception as e:
            print(f"Failed on {url}: {e}")
            
    print("\n--- RESULTS ---")
    print("Sources found:", sources)
    print("Sizes found:", sizes)
