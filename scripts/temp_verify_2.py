import urllib.request
import json
import ssl

ctx = ssl.create_default_context()
ctx.check_hostname = False
ctx.verify_mode = ssl.CERT_NONE

def head_url(url):
    req = urllib.request.Request(url, method='HEAD', headers={'User-Agent': 'Mozilla/5.0'})
    try:
        with urllib.request.urlopen(req, context=ctx, timeout=10) as response:
            return response.status
    except urllib.error.HTTPError as e:
        return e.code
    except Exception as e:
        return -1

def get_json(url):
    req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
    try:
        with urllib.request.urlopen(req, context=ctx, timeout=10) as response:
            return json.loads(response.read())
    except:
        return None

print("RSVQA Static Images:", head_url("https://zenodo.org/records/6344333/files/Images_LR.zip"))
print("RSVQA Static Questions:", head_url("https://zenodo.org/records/6344333/files/Questions_LR.zip"))

cdvqa_sample = get_json("https://raw.githubusercontent.com/YZHJessica/CDVQA/main/Test_images.json")
if cdvqa_sample:
    print("CDVQA Image ID Example:", cdvqa_sample[0] if len(cdvqa_sample) > 0 else "Empty")
cdvqa_q_sample = get_json("https://raw.githubusercontent.com/YZHJessica/CDVQA/main/Test_questions.json")
if cdvqa_q_sample:
    print("CDVQA Question Example:", cdvqa_q_sample[0] if len(cdvqa_q_sample) > 0 else "Empty")

def get_html(url):
    req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
    try:
        with urllib.request.urlopen(req, context=ctx, timeout=10) as response:
            return response.read().decode('utf-8')
    except:
        return ""
        
levir_html = get_html("https://justchenhao.github.io/LEVIR/")
print("LEVIR Train 445:", "445" in levir_html)
print("LEVIR Val 64:", "64" in levir_html)
print("LEVIR Test 128:", "128" in levir_html)
print("LEVIR Google Drive:", "drive.google.com" in levir_html)

