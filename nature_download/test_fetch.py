import re

from bs4 import BeautifulSoup
import requests

def estimate_panels(caption: str) -> int:
    import re
    if not caption:
        return 1
    # look for (a), (b), (c) ... (h)
    matches = re.findall(r'\(([a-z])\)', caption.lower())
    if not matches:
        return 1
    # Count distinct letters
    letters = set(matches)
    return len(letters)

url = "https://www.nature.com/articles/s41467-024-47987-x/figures/1"
r = requests.get(url)
soup = BeautifulSoup(r.text, "html.parser")

cap_el = soup.find("figcaption")
t = ""
if cap_el:
    t = cap_el.get_text(" ", strip=True)

print("Text found: ", t)
print("Panels estimated: ", estimate_panels(t))

