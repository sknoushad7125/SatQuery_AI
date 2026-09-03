import urllib.request
import urllib.error
import json
import ssl

def verify_url(url, expected_json=False):
    """Verifies a URL is accessible without downloading the payload."""
    try:
        ctx = ssl.create_default_context()
        ctx.check_hostname = False
        ctx.verify_mode = ssl.CERT_NONE
        
        req = urllib.request.Request(url, method='HEAD')
        if expected_json:
            req = urllib.request.Request(url, method='GET')
            
        with urllib.request.urlopen(req, timeout=10, context=ctx) as response:
            status = response.status
            
            if expected_json:
                data = json.loads(response.read().decode('utf-8'))
                return status == 200, "Success", data
            return status == 200, "Success", None
    except urllib.error.HTTPError as e:
        return False, f"HTTP Error {e.code}", None
    except Exception as e:
        return False, str(e), None

def report_verification(dataset, source_verified, download_verified, split_verified, schema_verified, blocker):
    print(f"VERIFY|{dataset}|{'Yes' if source_verified else 'No'}|{'Yes' if download_verified else 'No'}|{'Yes' if split_verified else 'No'}|{'Yes' if schema_verified else 'No'}|{blocker}")
