import requests
import json
import time

proxies = {"http": "http://127.0.0.1:8118", "https": "http://127.0.0.1:8118"}
url = "https://api.crossref.org/works?query=cancer&filter=type:journal-article&rows=10"
for i in range(5):
    try:
        r = requests.get(url, proxies=proxies, timeout=10)
        print("Success:", r.status_code)
    except Exception as e:
        print("Error:", type(e).__name__, e)
    time.sleep(1)
