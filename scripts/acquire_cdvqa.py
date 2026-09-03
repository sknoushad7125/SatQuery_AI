import argparse
import urllib.request
import json
import ssl

ctx = ssl.create_default_context()
ctx.check_hostname = False
ctx.verify_mode = ssl.CERT_NONE

def verify():
    url = "https://raw.githubusercontent.com/YZHJessica/CDVQA/main/Test_images.json"
    req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
    try:
        with urllib.request.urlopen(req, context=ctx, timeout=10) as response:
            data = json.loads(response.read())
            if "images" in data and len(data["images"]) > 0:
                print(f"CDVQA|VERIFIED|Image mapping format: {data['images'][0].get('file_name')}")
            else:
                print("CDVQA|FAILED|Schema mismatch")
    except Exception as e:
        print(f"CDVQA|FAILED|{str(e)}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--verify", action="store_true")
    if parser.parse_args().verify:
        verify()
