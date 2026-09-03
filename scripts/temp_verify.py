import urllib.request
import json
import ssl

ctx = ssl.create_default_context()
ctx.check_hostname = False
ctx.verify_mode = ssl.CERT_NONE

def fetch_json(url):
    req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
    try:
        with urllib.request.urlopen(req, context=ctx, timeout=15) as response:
            return response.status, json.loads(response.read())
    except urllib.error.HTTPError as e:
        return e.code, None
    except Exception as e:
        return -1, str(e)

print("--- RSVQA ---")
print(fetch_json("https://zenodo.org/api/records/6344333"))

print("\n--- CDVQA ---")
print(fetch_json("https://api.github.com/repos/YZHJessica/CDVQA/git/trees/main?recursive=1"))

print("\n--- BigEarthNet ---")
print(fetch_json("https://datasets-server.huggingface.co/info?dataset=BIFOLD-BigEarthNetv2-0"))

print("\n--- VRSBench ---")
print(fetch_json("https://datasets-server.huggingface.co/info?dataset=xiang709/VRSBench"))

