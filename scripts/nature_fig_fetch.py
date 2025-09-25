#!/usr/bin/env python3
import argparse
import importlib
import json
import os
import re
from urllib.parse import urljoin
import sys
import time
from pathlib import Path


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

UA = "PheroViz-NatureFigure/0.1 (+authorized; no-evasion; contact=local)"


def polite_get(url: str, timeout=30, sleep=1.0, max_retries=3):
    headers = {
        "User-Agent": UA,
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,*/*;q=0.8",
        "Accept-Language": "en-US,en;q=0.8,zh;q=0.7",
        "Connection": "close",
    }
    attempt = 0
    backoff = sleep if sleep else 0.5
    last_exc = None
    while attempt < max_retries:
        attempt += 1
        try:
            r = requests.get(url, headers=headers, timeout=timeout)
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


def parse_article_id_and_fig(url: str):
    # returns (article_id, figure_no or None)
    m = re.search(r"/articles/([^/]+)(?:/figures/(\d+))?", url)
    if not m:
        return ("unknown", None)
    aid = m.group(1)
    fno = m.group(2)
    return (aid, int(fno) if fno else None)


def pick_largest_src(soup: BeautifulSoup, base_url: str):
    # prefer og:image; else first figure img; choose largest srcset entry
    og = soup.find("meta", attrs={"property": "og:image"})
    if og and og.get("content"):
        return urljoin(base_url, og["content"].strip())
    img = None
    # nature figure page usually has figure or picture
    fig = soup.find("figure")
    if fig:
        # picture > source[srcset]
        src = fig.find("source")
        if src and src.get("srcset"):
            entries = [x.strip() for x in src["srcset"].split(",") if x.strip()]
            # take last as largest
            if entries:
                last = entries[-1].split()[0]
                return urljoin(base_url, last)
        img = fig.find("img")
    if not img:
        img = soup.find("img")
    if img and img.get("src"):
        return urljoin(base_url, img["src"].strip())
    return None


def extract_caption(soup: BeautifulSoup):
    # try figcaption
    cap_el = soup.find("figcaption")
    if cap_el:
        text = cap_el.get_text(" ", strip=True)
        if text:
            return text
    # try common classnames
    cap_div = soup.find(class_=re.compile(r"c-figure__caption|figure__caption|caption"))
    if cap_div:
        text = cap_div.get_text(" ", strip=True)
        if text:
            return text
    # data-test attributes that include caption
    cap_dt = soup.find(attrs={"data-test": re.compile(r"caption", re.I)})
    if cap_dt:
        text = cap_dt.get_text(" ", strip=True)
        if text:
            return text
    # meta fallbacks
    ogd = soup.find("meta", attrs={"property": "og:description"})
    if ogd and ogd.get("content"):
        return ogd["content"].strip()
    md = soup.find("meta", attrs={"name": "description"})
    if md and md.get("content"):
        return md["content"].strip()
    # JSON-LD fallback
    for sc in soup.find_all("script", attrs={"type": "application/ld+json"}):
        try:
            data = json.loads(sc.string or "{}")
        except Exception:
            continue
        def walk(obj):
            if isinstance(obj, dict):
                for k, v in obj.items():
                    if k in ("caption", "description") and isinstance(v, str) and v.strip():
                        return v.strip()
                    got = walk(v)
                    if got:
                        return got
            elif isinstance(obj, list):
                for it in obj:
                    got = walk(it)
                    if got:
                        return got
            return None
        found = walk(data)
        if found:
            return found
    return ""


def download_binary(url: str, out_path: Path, timeout=30, sleep=1.0, max_retries=3):
    headers = {"User-Agent": UA, "Accept": "*/*", "Connection": "close"}
    attempt = 0
    backoff = sleep if sleep else 0.5
    last_exc = None
    while attempt < max_retries:
        attempt += 1
        try:
            r = requests.get(url, headers=headers, timeout=timeout)
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
            print(f"  [retry] {url} failed ({e}); waiting {wait:.1f}s")
            time.sleep(wait)
    if last_exc:
        raise last_exc


def main():
    p = argparse.ArgumentParser(description="Fetch Nature.com figure image and caption (with authorization)")
    p.add_argument("--url", required=True, help="Nature article or figure URL, e.g., https://www.nature.com/articles/xxx/figures/1")
    p.add_argument("--out", default="outputs/nature_figs", help="Output base directory")
    p.add_argument("--sleep", type=float, default=1.0, help="Seconds to sleep between requests")
    p.add_argument("--timeout", type=float, default=30, help="Per-request timeout seconds")
    p.add_argument("--max-retries", type=int, default=3, help="Max retries per request")
    args = p.parse_args()

    url = args.url
    aid, fno = parse_article_id_and_fig(url)
    print(f"[info] Article: {aid} | Figure: {fno if fno else 'all/unknown'}")

    r = polite_get(url, timeout=args.timeout, sleep=args.sleep, max_retries=args.max_retries)
    html = r.text
    soup = BeautifulSoup(html, "html.parser")

    img_url = pick_largest_src(soup, r.url)
    caption = extract_caption(soup)

    outdir = Path(args.out) / aid
    outdir.mkdir(parents=True, exist_ok=True)
    meta = {
        "source_url": url,
        "article_id": aid,
        "figure_no": fno,
        "image_url": img_url,
        "caption": caption,
    }
    # write meta json
    with (outdir / (f"figure_{fno or 'page'}_meta.json")).open("w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)

    if img_url:
        # derive extension
        ext = ".jpg"
        m = re.search(r"\.(png|jpg|jpeg|gif|webp)(?:\?|$)", img_url, re.I)
        if m:
            ext = "." + m.group(1).lower().replace("jpeg", "jpg")
        out_img = outdir / f"figure_{fno or 'page'}{ext}"
        saved = download_binary(img_url, out_img, timeout=args.timeout, sleep=args.sleep, max_retries=args.max_retries)
        print(f"[done] Image saved: {saved}")
    else:
        print("[warn] No image URL found on the page")

    if caption:
        with (outdir / (f"figure_{fno or 'page'}_caption.txt")).open("w", encoding="utf-8") as f:
            f.write(caption)
        print("[done] Caption saved.")
    else:
        print("[warn] No caption text found.")


if __name__ == "__main__":
    main()
