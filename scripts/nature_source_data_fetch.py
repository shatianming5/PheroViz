#!/usr/bin/env python3
import argparse
import importlib
import json
import os
import re
import sys
import time
from pathlib import Path
from urllib.parse import urljoin, urlparse, unquote


def ensure_package(module_name: str, pip_name: str | None = None):
    try:
        return importlib.import_module(module_name)
    except ImportError:
        to_install = pip_name or module_name
        print(f"[setup] Installing missing dependency: {to_install} ...", flush=True)
        import subprocess
        subprocess.check_call([sys.executable, "-m", "pip", "install", to_install])
        return importlib.import_module(module_name)


requests = ensure_package("requests")
bs4 = ensure_package("bs4", "beautifulsoup4")
from bs4 import BeautifulSoup  # type: ignore


UA = "PheroViz-NatureSourceData/0.2 (+authorized; no-evasion; contact=local)"


def polite_get(url: str, timeout=30, sleep=1.0, max_retries=3, headers: dict | None = None):
    hdrs = {
        "User-Agent": UA,
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
        "Accept-Language": "en-US,en;q=0.8,zh;q=0.7",
        "Connection": "close",
    }
    if headers:
        hdrs.update(headers)
    attempt = 0
    backoff = sleep or 0.5
    last_exc = None
    while attempt < max_retries:
        attempt += 1
        try:
            r = requests.get(url, headers=hdrs, timeout=timeout)
            if sleep:
                time.sleep(sleep)
            r.raise_for_status()
            return r
        except Exception as e:
            last_exc = e
            wait = backoff * (2 ** (attempt - 1))
            print(f"  [retry] {url} failed ({e}); waiting {wait:.1f}s")
            time.sleep(wait)
    if last_exc:
        raise last_exc


def parse_article_id(url: str) -> str:
    m = re.search(r"/articles/([^/#?]+)", url)
    return m.group(1) if m else "unknown"


def find_source_data_links(soup: BeautifulSoup, base_url: str, section_id: str | None, text_filter: str | None):
    links = []
    container = soup
    if section_id:
        sec = soup.find(id=section_id)
        if sec:
            container = sec

    rx = re.compile(r"source\s*data", re.I)
    for a in container.find_all("a"):
        label = (a.get_text(" ", strip=True) or "").strip()
        if not label:
            continue
        if not rx.search(label):
            continue
        if text_filter and (text_filter.lower() not in label.lower()):
            continue
        href = a.get("href")
        if not href:
            continue
        abs_url = urljoin(base_url, href)
        links.append({"label": label, "url": abs_url})
    # de-dup by URL
    seen = set()
    uniq = []
    for L in links:
        if L["url"] not in seen:
            seen.add(L["url"])
            uniq.append(L)
    return uniq


def filename_from_url(url: str) -> str:
    path = urlparse(url).path
    name = os.path.basename(path)
    return unquote(name) or "source_data.bin"


def download_file(file_url: str, out_path: Path, referer: str, timeout=30, sleep=1.0, max_retries=3):
    headers = {"User-Agent": UA, "Accept": "*/*", "Connection": "close", "Referer": referer}
    attempt = 0
    backoff = sleep or 0.5
    last_exc = None
    while attempt < max_retries:
        attempt += 1
        try:
            r = requests.get(file_url, headers=headers, timeout=timeout)
            if sleep:
                time.sleep(sleep)
            r.raise_for_status()
            out_path.parent.mkdir(parents=True, exist_ok=True)
            with out_path.open("wb") as f:
                f.write(r.content)
            return str(out_path)
        except Exception as e:
            last_exc = e
            wait = backoff * (2 ** (attempt - 1))
            print(f"  [retry] {file_url} failed ({e}); waiting {wait:.1f}s")
            time.sleep(wait)
    if last_exc:
        raise last_exc


def ensure_dir(p: Path):
    p.mkdir(parents=True, exist_ok=True)


def sanitize(s: str) -> str:
    return re.sub(r"[^a-zA-Z0-9_.\-]+", "_", s).strip("_.-") or "source_data"


def upsert_json_list(path: Path, item: dict, key: str):
    data = []
    if path.exists():
        try:
            data = json.loads(path.read_text(encoding="utf-8") or "[]")
        except Exception:
            data = []
    existing_keys = {str(d.get(key)) for d in data}
    if str(item.get(key)) not in existing_keys:
        data.append(item)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


def main():
    p = argparse.ArgumentParser(description="Fetch Nature.com Source data attachments (with authorization)")
    p.add_argument("--url", required=True, help="Nature article URL, e.g., https://www.nature.com/articles/xxx#Sec71")
    p.add_argument("--out", default="outputs/nature_source_data", help="Output base directory")
    p.add_argument("--section-id", default=None, help="Optional section id to narrow search, e.g., Sec71")
    p.add_argument("--filter", default=None, help="Optional substring filter in link label, e.g., 'Fig. 4' or 'Extended Data Fig. 7'")
    p.add_argument("--sleep", type=float, default=1.0, help="Seconds to sleep between requests")
    p.add_argument("--timeout", type=float, default=30, help="Per-request timeout seconds")
    p.add_argument("--max-retries", type=int, default=3, help="Max retries per request")
    args = p.parse_args()

    url = args.url
    art_id = parse_article_id(url)
    print(f"[info] Article: {art_id} | section: {args.section_id or 'all'} | filter: {args.filter or 'none'}")

    r = polite_get(url, timeout=args.timeout, sleep=args.sleep, max_retries=args.max_retries)
    soup = BeautifulSoup(r.text, "html.parser")

    links = find_source_data_links(soup, r.url, args.section_id, args.filter)
    print(f"[info] Source data links found: {len(links)}")

    base = Path(args.out) / art_id
    sd_dir = base / "source_data"
    meta_dir = base / "meta"
    ensure_dir(sd_dir)
    ensure_dir(meta_dir)
    manifest = []
    for i, L in enumerate(links, 1):
        label = L["label"]
        file_url = L["url"]
        fname_url = filename_from_url(file_url)
        ext = Path(fname_url).suffix
        label_safe = sanitize(label)
        fname = label_safe + (ext or "")
        save_to = sd_dir / fname
        print(f"  [{i}/{len(links)}] {label} -> {fname}")
        try:
            saved = download_file(file_url, save_to, referer=r.url, timeout=args.timeout, sleep=args.sleep, max_retries=args.max_retries)
            entry = {"label": label, "url": file_url, "saved_as": str(save_to), "orig_name": fname_url}
            manifest.append(entry)
            upsert_json_list(meta_dir / "source_data.json", entry, key="label")
        except Exception as e:
            print(f"  [warn] download failed: {file_url} ({e})")
            entry = {"label": label, "url": file_url, "error": str(e), "orig_name": fname_url}
            manifest.append(entry)
            upsert_json_list(meta_dir / "source_data.json", entry, key="label")

    with (meta_dir / "_source_data_manifest.json").open("w", encoding="utf-8") as f:
        json.dump({"article_url": url, "links": manifest}, f, ensure_ascii=False, indent=2)
    print(f"[done] Saved manifest and files under {base}")


if __name__ == "__main__":
    main()
