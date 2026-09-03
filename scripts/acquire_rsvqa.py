import argparse
import urllib.request
import ssl

ctx = ssl.create_default_context()
ctx.check_hostname = False
ctx.verify_mode = ssl.CERT_NONE

def verify():
    url = "https://zenodo.org/api/records/6344333"
    req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
    try:
        with urllib.request.urlopen(req, context=ctx, timeout=10) as response:
            print("RSVQA-LR|VERIFIED|6344333 found")
    except urllib.error.HTTPError as e:
        if e.code == 403:
            print("RSVQA-LR|BLOCKED|Zenodo bot protection (403). Requires manual download or token.")
        else:
            print(f"RSVQA-LR|FAILED|{e.code}")
    except Exception as e:
        print(f"RSVQA-LR|FAILED|{str(e)}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--verify", action="store_true")
    if parser.parse_args().verify:
        verify()
