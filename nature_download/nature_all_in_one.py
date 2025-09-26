#!/usr/bin/env python3
"""
Nature family search + authorized content fetch (all-in-one)

Subcommands:
- search: Crossref + Europe PMC/PMC åˆè§„æ£€ç´¢ï¼Œå¯¼å‡º JSONL/CSV
- postfetch: å¯¹æ£€ç´¢ç»“æžœé€ç¯‡æŠ“å–ï¼ˆå›¾åƒ+caption + Source dataï¼‰ï¼Œå§‹ç»ˆâ€œæŠ“å–å…¨éƒ¨â€
- fig: æŠ“å– nature.com å•ä¸ªå›¾é¡µï¼ˆå·²æŽˆæƒå‰æï¼‰
- source: æŠ“å– nature.com æ–‡ç« é¡µ Source dataï¼ˆå·²æŽˆæƒå‰æï¼‰
- auto: å¤šå…³é”®è¯æ‰¹é‡æœç´¢å¹¶éšåŽ postfetchï¼ˆå·²æŽˆæƒå‰æï¼‰

è¾“å‡ºç»“æž„ï¼ˆç»Ÿä¸€ï¼‰ï¼š<out>/<article-id>/{figures, source_data, meta}
"""
from __future__ import annotations

import argparse
import csv
import importlib
import json
import mimetypes
import os
import re
import subprocess
import sys
import time
from email.message import Message
from pathlib import Path
from typing import Any
from urllib.parse import urljoin, urlparse, unquote
from uuid import uuid4
import shutil
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed, wait, FIRST_COMPLETED


def ensure_package(module_name: str, pip_name: str | None = None):
    try:
        return importlib.import_module(module_name)
    except ImportError:
        to_install = pip_name or module_name
        print(f"[setup] Installing missing dependency: {to_install} ...", flush=True)
        subprocess.check_call([sys.executable, "-m", "pip", "install", to_install])
        return importlib.import_module(module_name)


requests = ensure_package("requests")
bs4 = ensure_package("bs4", "beautifulsoup4")
from bs4 import BeautifulSoup  # type: ignore
try:
    ensure_package("rich")
    from rich.console import Console
    from rich.progress import Progress, SpinnerColumn, BarColumn, TextColumn, TimeElapsedColumn
    console = Console()
except Exception:
    console = None


API_USER_AGENT = "PheroViz-NatureAllInOne/1.0 (+compliant; authorized when required)"


def safe_console(text: str) -> str:
    try:
        enc = sys.stdout.encoding or "utf-8"
        return text.encode(enc, errors="replace").decode(enc, errors="replace")
    except Exception:
        try:
            return text.encode("ascii", errors="replace").decode("ascii")
        except Exception:
            return "<unprintable>"


def polite_get(url: str, params=None, timeout=30, sleep=1.0, max_retries=3, headers: dict[str, str] | None = None):
    hdrs = {
        "User-Agent": API_USER_AGENT,
        "Accept": "*/*",
        "Connection": "close",
    }
    if headers:
        hdrs.update(headers)
    attempt = 0
    backoff = sleep if sleep else 0.5
    while True:
        attempt += 1
        try:
            resp = requests.get(url, params=params, headers=hdrs, timeout=timeout)
            if sleep:
                time.sleep(sleep)
            resp.raise_for_status()
            return resp
        except Exception as e:
            if attempt >= max_retries:
                raise
            wait = backoff * (2 ** (attempt - 1))
            print(safe_console(f"  [retry] {url} failed ({e}); waiting {wait:.1f}s"))
            time.sleep(wait)


# ---------- Search (Crossref + Europe PMC) ----------


def is_nature_family(container_titles) -> bool:
    if not container_titles:
        return False
    titles = [container_titles] if isinstance(container_titles, str) else container_titles
    for t in titles:
        if not t:
            continue
        tt = t.strip()
        if tt == "Nature" or tt.lower().startswith("nature "):
            return True
    return False


def crossref_search(query: str, rows: int = 20, mailto: str | None = None, sleep=1.0, timeout=30, max_retries=3, family_bias=True):
    base = "https://api.crossref.org/works"
    params = {
        "query": query,
        "filter": "type:journal-article",
        "rows": rows,
    }
    if family_bias:
        params["query.container-title"] = "Nature"
    if mailto:
        params["mailto"] = mailto
    r = polite_get(base, params=params, sleep=sleep, timeout=timeout, max_retries=max_retries)
    data = r.json()
    items = data.get("message", {}).get("items", [])
    return [it for it in items if is_nature_family(it.get("container-title"))]


def crossref_cursor_stream(
    query: str,
    total_max: int,
    mailto: str | None = None,
    sleep: float = 1.0,
    timeout: float = 30,
    max_retries: int = 3,
    family_bias: bool = True,
    page_rows: int = 1000,
):
    """
    Yield items from Crossref using cursor pagination to go beyond single-request limits.
    """
    base = "https://api.crossref.org/works"
    fetched = 0
    cursor = "*"
    page_rows = min(max(1, int(page_rows)), 1000)
    while fetched < total_max:
        remaining = total_max - fetched
        rows = min(page_rows, remaining)
        params = {
            "query": query,
            "filter": "type:journal-article",
            "rows": rows,
            "cursor": cursor,
        }
        if family_bias:
            params["query.container-title"] = "Nature"
        if mailto:
            params["mailto"] = mailto
        try:
            r = polite_get(base, params=params, sleep=sleep, timeout=timeout, max_retries=max_retries)
        except requests.HTTPError as exc:
            status = getattr(exc.response, "status_code", None)
            if status == 404:
                msg = safe_console(f"[warn] Crossref cursor 404 for query {query!r}; skipping remainder")
                if console:
                    console.log(msg)
                else:
                    print(msg)
                break
            raise
        data = r.json()
        items = data.get("message", {}).get("items", [])
        for it in items:
            if is_nature_family(it.get("container-title")):
                yield it
                fetched += 1
                if fetched >= total_max:
                    break
        next_cursor = data.get("message", {}).get("next-cursor") or data.get("message", {}).get("next_cursor")
        if not next_cursor:
            break
        cursor = next_cursor


def europe_pmc_by_doi(doi: str, sleep=1.0, timeout=30, max_retries=3):
    url = "https://www.ebi.ac.uk/europepmc/webservices/rest/search"
    params = {"query": f"DOI:{doi}", "format": "json", "pageSize": 1}
    r = polite_get(url, params=params, sleep=sleep, timeout=timeout, max_retries=max_retries)
    data = r.json()
    results = data.get("resultList", {}).get("result", [])
    return results[0] if results else None


def fetch_pmc_figure_urls(pmcid: str, sleep=1.0, timeout=30, max_retries=3):
    base = f"https://pmc.ncbi.nlm.nih.gov/articles/{pmcid}/"
    r = polite_get(base, sleep=sleep, timeout=timeout, max_retries=max_retries)
    soup = BeautifulSoup(r.text, "html.parser")
    urls: list[str] = []
    for fig in soup.find_all("figure"):
        for img in fig.find_all("img"):
            src = img.get("src") or img.get("data-src")
            if not src:
                continue
            if src.startswith("//"):
                full = "https:" + src
            elif src.startswith("/"):
                full = "https://pmc.ncbi.nlm.nih.gov" + src
            elif src.startswith("http"):
                full = src
            else:
                full = base + src
            urls.append(full)
    seen = set()
    unique = []
    for u in urls:
        if u not in seen:
            seen.add(u)
            unique.append(u)
    return unique


def sanitize(s: str) -> str:
    return re.sub(r"[^a-zA-Z0-9_.\-]+", "_", s).strip("_.-") or "item"


def write_jsonl(path: Path, rows: list[dict]):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def write_csv(path: Path, rows: list[dict], fieldnames: list[str]):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k, "") for k in fieldnames})


def load_processed_set(path: Path) -> set[str]:
    s: set[str] = set()
    if path.exists():
        try:
            for line in path.read_text(encoding="utf-8").splitlines():
                line = line.strip()
                if line:
                    s.add(line)
        except Exception:
            pass
    return s


def append_processed(path: Path, article_id: str):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(article_id + "\n")


def append_skipped(path: Path, article_id: str, reason: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    safe_reason = reason.strip() or "unknown"
    line = f"{article_id}\t{safe_reason}\n"
    with path.open("a", encoding="utf-8") as f:
        f.write(line)




def append_jsonl(path: Path, obj: dict):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(obj, ensure_ascii=False) + "\n")


def format_authors_crossref(author_list) -> str:
    try:
        parts = []
        for a in author_list or []:
            given = (a.get("given") or "").strip()
            family = (a.get("family") or "").strip()
            name = (family + ", " + given).strip(", ") if (given or family) else (a.get("name") or "")
            if name:
                parts.append(name)
        return "; ".join(parts)
    except Exception:
        return ""


def merge_append(records: list[dict], existing_path: Path) -> list[dict]:
    if not existing_path.exists():
        return records
    seen = set()
    merged = []
    try:
        with existing_path.open("r", encoding="utf-8") as f:
            for line in f:
                try:
                    obj = json.loads(line)
                    doi = (obj.get("doi") or "").lower()
                    if doi and doi not in seen:
                        seen.add(doi)
                        merged.append(obj)
                except Exception:
                    continue
    except Exception:
        pass
    for r in records:
        doi = (r.get("doi") or "").lower()
        if doi and doi not in seen:
            seen.add(doi)
            merged.append(r)
    return merged


def cmd_search(args):
    print(f"[info] Query: {safe_console(args.query)}")
    print(f"[info] Max: {args.max} | Sleep: {args.sleep}s")
    # Support cursor pagination when requesting >1000 items
    if args.max and args.max > 1000:
        items_iter = crossref_cursor_stream(
            args.query,
            total_max=args.max,
            mailto=args.mailto,
            sleep=args.sleep,
            timeout=args.timeout,
            max_retries=args.max_retries,
            family_bias=not args.no_family_bias,
            page_rows=1000,
        )
        items = list(items_iter)
    else:
        items = crossref_search(
            args.query,
            rows=args.max,
            mailto=args.mailto,
            sleep=args.sleep,
            timeout=args.timeout,
            max_retries=args.max_retries,
            family_bias=not args.no_family_bias,
        )
    print(f"[info] Crossref filtered results (Nature family): {len(items)}")
    records: list[dict[str, Any]] = []
    outdir = Path(args.out)
    outdir.mkdir(parents=True, exist_ok=True)
    for i, it in enumerate(items, 1):
        doi = it.get("DOI")
        title_list = it.get("title") or []
        title = title_list[0] if title_list else ""
        container = (it.get("container-title") or [""])[0]
        issued = it.get("issued", {}).get("date-parts", [[None]])[0]
        year = issued[0] if issued else None
        url = it.get("URL")
        abstract = it.get("abstract")

        epmc = europe_pmc_by_doi(doi, sleep=args.sleep, timeout=args.timeout, max_retries=args.max_retries)
        pmcid = epmc.get("pmcid") if epmc else None
        abstract_epmc = epmc.get("abstractText") if epmc else None
        is_oa = bool(epmc.get("isOpenAccess") or pmcid) if epmc else False
        pmc_url = f"https://pmc.ncbi.nlm.nih.gov/articles/{pmcid}/" if pmcid else None
        pmid = epmc.get("pmid") if epmc else None

        final_abstract = abstract_epmc or abstract or ""
        authors = format_authors_crossref(it.get("author")) or (epmc.get("authorString") if epmc else "")
        rec = {
            "doi": doi,
            "title": title,
            "journal": container,
            "year": year,
            "url": url,
            "pmcid": pmcid,
            "pmc_url": pmc_url,
            "pmid": pmid,
            "is_open_access": is_oa,
            "abstract": final_abstract,
            "authors": authors,
        }
        print(f"[{i}/{len(items)}] {safe_console(title)[:80]}...")
        print(f"      DOI: {doi} | Journal: {safe_console(container)} | Year: {year}")
        records.append(rec)

    jsonl_path = outdir / "articles.jsonl"
    if args.append and jsonl_path.exists():
        write_jsonl(jsonl_path, merge_append(records, jsonl_path))
    else:
        write_jsonl(jsonl_path, records)
    fields = ["doi", "title", "journal", "year", "url", "pmcid", "pmc_url", "pmid", "is_open_access", "abstract", "authors"]
    write_csv(outdir / "articles.csv", records, fields)
    print(f"[done] Saved {len(records)} records to {outdir}")


# ---------- Authorized nature.com figure page (image + caption) ----------


def is_likely_image_url(u: str) -> bool:
    try:
        p = urlparse(u)
        if p.scheme not in ("http", "https"):
            return False
        host = (p.netloc or "").lower()
        if any(bad in host for bad in ["doubleclick.net", "googletagservices.com", "gampad"]):
            return False
        path = (p.path or "").lower()
        if any(path.endswith(ext) for ext in (".png", ".jpg", ".jpeg", ".gif", ".webp")):
            return True
        if "media.springernature.com" in host:
            return True
        return False
    except Exception:
        return False


def normalize_img_url(u: str) -> str:
    u = u.strip()
    if u.startswith("//"):
        u = "https:" + u
    m = re.search(r"(https?://images\.nature\.com/[^\s\"]+)", u)
    if m:
        return m.group(1)
    return u


def pick_largest_src(soup: BeautifulSoup, base_url: str):
    og = soup.find("meta", attrs={"property": "og:image"})
    if og and og.get("content"):
        cand = normalize_img_url(urljoin(base_url, og["content"].strip()))
        if is_likely_image_url(cand):
            return cand
    fig = soup.find("figure")
    if fig:
        src = fig.find("source")
        if src and src.get("srcset"):
            entries = [x.strip() for x in src["srcset"].split(",") if x.strip()]
            for entry in reversed(entries):
                u = normalize_img_url(urljoin(base_url, entry.split()[0]))
                if is_likely_image_url(u):
                    return u
        img = fig.find("img")
    else:
        img = soup.find("img")
    if img:
        src_attr = img.get("src") or img.get("data-src") or img.get("data-original")
        if src_attr:
            cand = normalize_img_url(urljoin(base_url, src_attr.strip()))
            if is_likely_image_url(cand):
                return cand
    # Scan anchors for potential high-res downloads
    best = None
    def score(u: str) -> int:
        s = 0
        uu = u.lower()
        if any(k in uu for k in ("original", "download", "full")):
            s += 5
        if uu.endswith(".tif") or ".tif?" in uu:
            s += 4
        elif uu.endswith(".png") or ".png?" in uu:
            s += 3
        elif uu.endswith(".jpg") or uu.endswith(".jpeg") or ".jpg?" in uu or ".jpeg?" in uu:
            s += 2
        elif uu.endswith(".webp") or ".webp?" in uu:
            s += 1
        return s
    for a in soup.find_all("a"):
        href = a.get("href")
        if not href:
            continue
        u = normalize_img_url(urljoin(base_url, href))
        if is_likely_image_url(u):
            sc = score(u)
            if not best or sc > best[0]:
                best = (sc, u)
    if best:
        return best[1]
    return None


def extract_caption(soup: BeautifulSoup):
    cap_el = soup.find("figcaption")
    if cap_el:
        t = cap_el.get_text(" ", strip=True)
        if t:
            return t
    cap_div = soup.find(class_=re.compile(r"c-figure__caption|figure__caption|caption"))
    if cap_div:
        t = cap_div.get_text(" ", strip=True)
        if t:
            return t
    cap_dt = soup.find(attrs={"data-test": re.compile(r"caption", re.I)})
    if cap_dt:
        t = cap_dt.get_text(" ", strip=True)
        if t:
            return t
    ogd = soup.find("meta", attrs={"property": "og:description"})
    if ogd and ogd.get("content"):
        return ogd["content"].strip()
    md = soup.find("meta", attrs={"name": "description"})
    if md and md.get("content"):
        return md["content"].strip()
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


def ensure_dir(p: Path):
    p.mkdir(parents=True, exist_ok=True)


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


def download_binary(url: str, out_path: Path, referer: str | None = None, timeout=30, sleep=1.0, max_retries=3, *, return_meta: bool = False):
    headers = {"User-Agent": API_USER_AGENT, "Accept": "*/*", "Connection": "close"}
    if referer:
        headers["Referer"] = referer
    attempt = 0
    backoff = sleep if sleep else 0.5
    while True:
        attempt += 1
        try:
            resp = requests.get(url, headers=headers, timeout=(timeout, timeout), stream=True)
            resp.raise_for_status()
            expected = resp.headers.get("Content-Length")
            remote_name = parse_content_disposition_filename(resp.headers.get("Content-Disposition"))
            content_type = resp.headers.get("Content-Type")
            out_path.parent.mkdir(parents=True, exist_ok=True)
            tmp = out_path.with_suffix(out_path.suffix + ".part")
            total = 0
            with tmp.open("wb") as f:
                for chunk in resp.iter_content(chunk_size=8192):
                    if chunk:
                        f.write(chunk)
                        total += len(chunk)
            if expected and total != int(expected):
                raise IOError(f"incomplete download: {total} != {expected}")
            tmp.replace(out_path)
            if return_meta:
                return str(out_path), remote_name, content_type
            return str(out_path)
        except Exception as e:
            if attempt >= max_retries:
                raise
            wait = backoff * (2 ** (attempt - 1))
            print(safe_console(f"  [retry-img] {url} ({e}); waiting {wait:.1f}s"))
            time.sleep(wait)


def parse_article_id_and_fig(url: str):
    m = re.search(r"/articles/([^/]+)(?:/figures/(\d+))?", url)
    if not m:
        return ("unknown", None)
    return (m.group(1), int(m.group(2)) if m.group(2) else None)


def cmd_fig(args):
    url = args.url
    aid, fno = parse_article_id_and_fig(url)
    print(f"[info] Article: {aid} | Figure: {fno if fno else 'all/unknown'}")
    try:
        r = polite_get(url, timeout=args.timeout, sleep=args.sleep, max_retries=args.max_retries)
    except Exception as e:
        print(safe_console(f"[warn] figure page not available: {url} ({e})"))
        return
    soup = BeautifulSoup(r.text, "html.parser")
    img_url = pick_largest_src(soup, r.url)
    caption = extract_caption(soup)

    base = Path(args.out) / aid
    figures_dir = base / "figures"
    meta_dir = base / "meta"
    ensure_dir(figures_dir)
    ensure_dir(meta_dir)

    fig_id = (fno if fno is not None else "page")
    fig_tag = f"fig_{fig_id:03d}" if isinstance(fig_id, int) else f"fig_{fig_id}"

    saved_img = None
    if img_url:
        ext = ".jpg"
        m = re.search(r"\.(png|jpg|jpeg|gif|webp)(?:\?|$)", img_url, re.I)
        if m:
            ext = "." + m.group(1).lower().replace("jpeg", "jpg")
        out_img = figures_dir / f"{fig_tag}{ext}"
        saved_img = download_binary(img_url, out_img, referer=r.url, timeout=args.timeout, sleep=args.sleep, max_retries=args.max_retries)
        print(f"[done] Image saved: {saved_img}")
    else:
        print("[warn] No image URL found on the page")

    # Only save caption if image exists; skip caption-only figures
    saved_cap = None
    if saved_img and caption:
        out_cap = figures_dir / f"{fig_tag}.txt"
        out_cap.write_text(caption, encoding="utf-8")
        saved_cap = str(out_cap)
        print("[done] Caption saved.")
    elif not caption:
        print("[warn] No caption text found.")

    # Only record figures with images
    if saved_img:
        entry = {"figure_tag": fig_tag, "figure_no": fno, "image_file": saved_img, "caption_file": saved_cap, "image_url": img_url, "source_url": url}
        upsert_json_list(meta_dir / "figures.json", entry, key="figure_tag")

    # return whether we found an image (skip caption-only)
    return bool(saved_img)


# ---------- Authorized nature.com Source data ----------


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
    base_parsed = urlparse(base_url)
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
        parsed = urlparse(abs_url)
        # Skip in-page anchors (same article URL with only fragment)
        if (
            parsed.fragment
            and parsed.scheme == base_parsed.scheme
            and parsed.netloc == base_parsed.netloc
            and parsed.path == base_parsed.path
            and not parsed.query
        ):
            continue
        links.append({"label": label, "url": abs_url})
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



def parse_content_disposition_filename(header: str | None) -> str | None:
    if not header:
        return None
    try:
        msg = Message()
        msg['content-disposition'] = header
        filename = msg.get_param('filename', header='content-disposition')
        if filename:
            return unquote(filename.strip())
        filename_star = msg.get_param('filename*', header='content-disposition')
        if filename_star:
            parts = filename_star.split("''", 1)
            if len(parts) == 2:
                filename_star = parts[1]
            return unquote(filename_star.strip())
    except Exception:
        pass
    match = re.search(r'filename\*?=\s*"?([^";]+)', header, re.I)
    if match:
        candidate = match.group(1)
        if "''" in candidate:
            candidate = candidate.split("''", 1)[1]
        return unquote(candidate.strip())
    return None


def guess_extension_from_type(content_type: str | None) -> str:
    if not content_type:
        return ""
    ctype = content_type.split(';', 1)[0].strip().lower()
    if not ctype:
        return ""
    ext = mimetypes.guess_extension(ctype)
    if ext in (None, "", ".bin") and ctype.startswith('text/'):
        return ".txt"
    return ext or ""


def cmd_source(args):
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
    manifest_path = meta_dir / "_source_data_manifest.json"
    json_path = meta_dir / "source_data.json"

    def cleanup_empty():
        if sd_dir.exists():
            shutil.rmtree(sd_dir, ignore_errors=True)
        if manifest_path.exists():
            manifest_path.unlink()
        if json_path.exists():
            json_path.unlink()

    if not links:
        print("[warn] No Source data links present; skip article")
        cleanup_empty()
        return False

    ensure_dir(sd_dir)
    ensure_dir(meta_dir)
    manifest = []
    saved_count = 0
    used_names: set[str] = set()

    def allocate_name(candidates: list[tuple[str, str]], fallback_stem: str, ext_hint: str) -> str:
        for stem, suffix in candidates:
            suffix = suffix or ext_hint
            stem = stem or fallback_stem
            attempt = stem + (suffix or "")
            counter = 2
            while attempt.lower() in used_names:
                attempt = f"{stem}_{counter}{suffix or ''}"
                counter += 1
            used_names.add(attempt.lower())
            return attempt
        suffix = ext_hint or ""
        stem = fallback_stem or "source_data"
        attempt = stem + suffix
        counter = 2
        while attempt.lower() in used_names:
            attempt = f"{stem}_{counter}{suffix}"
            counter += 1
        used_names.add(attempt.lower())
        return attempt

    for i, L in enumerate(links, 1):
        label = L["label"]
        file_url = L["url"]
        fname_url = filename_from_url(file_url)
        fallback_stem = f"source_data_{i:02d}"
        tmp_path = sd_dir / f".tmp_{uuid4().hex}"
        print(f"  [{i}/{len(links)}] {safe_console(label)} -> downloading...")
        try:
            saved_tmp, remote_name, content_type = download_binary(
                file_url,
                tmp_path,
                referer=r.url,
                timeout=args.timeout,
                sleep=args.sleep,
                max_retries=args.max_retries,
                return_meta=True,
            )
            tmp_file = Path(saved_tmp)
            ext_candidates = []
            for raw in (remote_name, fname_url):
                if raw:
                    ext = Path(raw).suffix
                    if ext:
                        ext_candidates.append(ext)
            type_ext = guess_extension_from_type(content_type)
            if type_ext:
                ext_candidates.append(type_ext)
            ext_hint = next((e for e in ext_candidates if e), "")
            candidate_pairs: list[tuple[str, str]] = []
            seen_pairs: set[tuple[str, str]] = set()

            def add_candidate(raw: str | None):
                if not raw:
                    return
                cleaned = sanitize(raw)
                if not cleaned:
                    return
                stem, suffix = os.path.splitext(cleaned)
                if not suffix and ext_hint:
                    suffix = ext_hint
                stem = stem or fallback_stem
                pair = (stem, suffix or "")
                if pair not in seen_pairs:
                    seen_pairs.add(pair)
                    candidate_pairs.append(pair)

            add_candidate(remote_name)
            add_candidate(fname_url)
            add_candidate(label)

            if not candidate_pairs:
                candidate_pairs.append((fallback_stem, ext_hint or ""))

            chosen_name = allocate_name(candidate_pairs, fallback_stem, ext_hint)
            final_path = sd_dir / chosen_name
            if final_path.exists():
                final_path.unlink()
            tmp_file.replace(final_path)
            print(f"      saved as {chosen_name}")
            entry = {"label": label, "url": file_url, "saved_as": str(final_path), "saved_name": chosen_name, "orig_name": fname_url, "content_name": remote_name, "content_type": content_type}
            manifest.append(entry)
            upsert_json_list(json_path, entry, key="label")
            saved_count += 1
        except Exception as e:
            if tmp_path.exists():
                try:
                    tmp_path.unlink()
                except Exception:
                    pass
            part_path = tmp_path.with_suffix(tmp_path.suffix + ".part")
            if part_path.exists():
                try:
                    part_path.unlink()
                except Exception:
                    pass
            entry = {"label": label, "url": file_url, "error": str(e), "saved_name": None, "orig_name": fname_url, "content_name": None, "content_type": None}
            manifest.append(entry)
            upsert_json_list(json_path, entry, key="label")

    if saved_count == 0:
        print("[warn] Source data downloads all failed; skip article")
        cleanup_empty()
        return False

    manifest_path.write_text(json.dumps({"article_url": url, "links": manifest}, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[done] Saved manifest and files under {base}")
    return True


# ---------- Post-fetch (always figures + source) ----------


def norm_article_url(url: str | None, doi: str | None) -> str | None:
    if url:
        m = re.search(r"https?://(?:www\.)?nature\.com/articles/([^/?#]+)", url)
        if m:
            aid = m.group(1)
            return f"https://www.nature.com/articles/{aid}"
    if doi and doi.lower().startswith("10.1038/"):
        suffix = doi.split("/", 1)[1]
        return f"https://www.nature.com/articles/{suffix}"
    return None


def cmd_postfetch(args):
    rows = []
    with Path(args.jsonl).open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except Exception:
                continue
    print(f"[info] Loaded {len(rows)} records from {args.jsonl}")
    if args.sort != "input":
        def year_key(x):
            y = x.get("year")
            try:
                return int(y) if y is not None else -10**9
            except Exception:
                return -10**9
        reverse = (args.sort == "year_desc")
        rows = sorted(rows, key=year_key, reverse=reverse)

    # Build task list with processed-file skipping
    processed_file_arg = getattr(args, "processed_file", None)
    processed_file = Path(processed_file_arg) if processed_file_arg else (Path(args.out) / "_processed.txt")
    processed = load_processed_set(processed_file)
    skipped_file = processed_file.with_name("_skipped.txt")
    tasks: list[str] = []
    for r in rows:
        art_url = norm_article_url(r.get("url"), r.get("doi"))
        if not art_url:
            continue
        aid = parse_article_id(art_url)
        if aid in processed:
            continue
        tasks.append(art_url)
        if args.max_articles and len(tasks) >= args.max_articles:
            break

    total = len(tasks)
    if total == 0:
        print("[info] No tasks to fetch (all processed or none found).")
        return

    workers = getattr(args, "workers", 1)
    if workers > 1:
        # Parallel with optional rich progress
        if console:
            with Progress(SpinnerColumn(spinner_name="line"), TextColumn("[progress.description]{task.description}"), BarColumn(), TextColumn("{task.completed}/{task.total}"), TimeElapsedColumn(), console=console) as progress:
                t = progress.add_task("Postfetch", total=total)
                with ProcessPoolExecutor(max_workers=workers) as ex:
                    futures = {ex.submit(postfetch_article, u, args.out, args.max_figs, getattr(args, "max_empty_figs", 2), args.sleep, args.timeout, args.max_retries): u for u in tasks}
                    for fut in as_completed(futures):
                        try:
                            aid, found = fut.result()
                        except Exception as e:
                            console.log(safe_console(f"[warn] postfetch failed: {e}"))
                            art = futures.get(fut)
                            if art:
                                aid = parse_article_id(art)
                                append_processed(processed_file, aid)
                                append_skipped(skipped_file, aid, "fetch-error")
                            continue
                        if found > 0:
                            append_processed(processed_file, aid)
                        else:
                            reason = "no-source-data" if found == -1 else "no-figures"
                            append_processed(processed_file, aid)
                            append_skipped(skipped_file, aid, reason)
                        progress.advance(t, 1)
        else:
            with ProcessPoolExecutor(max_workers=workers) as ex:
                futures = {ex.submit(postfetch_article, u, args.out, args.max_figs, getattr(args, "max_empty_figs", 2), args.sleep, args.timeout, args.max_retries): u for u in tasks}
                for fut in as_completed(futures):
                    try:
                        aid, found = fut.result()
                    except Exception as e:
                        print(safe_console(f"[warn] postfetch failed: {e}"))
                        art = futures.get(fut)
                        if art:
                            aid = parse_article_id(art)
                            append_processed(processed_file, aid)
                            append_skipped(skipped_file, aid, "fetch-error")
                        continue
                    if found > 0:
                        append_processed(processed_file, aid)
                    else:
                        reason = "no-source-data" if found == -1 else "no-figures"
                        append_processed(processed_file, aid)
                        append_skipped(skipped_file, aid, reason)
    else:
        for idx, u in enumerate(tasks, 1):
            aid = parse_article_id(u)
            print(f"[{idx}/{total}] Nature article: {aid}")
            found = postfetch_one(u, args.out, args.max_figs, args.sleep, args.timeout, args.max_retries, getattr(args, "max_empty_figs", 2))
            reason: str | None = None
            if found is None:
                reason = "fetch-error"
            elif found > 0:
                ok = cmd_source(argparse.Namespace(url=u, out=args.out, section_id=None, filter=None, sleep=args.sleep, timeout=args.timeout, max_retries=args.max_retries))
                if ok:
                    append_processed(processed_file, aid)
                else:
                    reason = "no-source-data"
            else:
                reason = "no-figures"
            if reason:
                base = Path(args.out) / aid
                if base.exists():
                    shutil.rmtree(base, ignore_errors=True)
                append_processed(processed_file, aid)
                append_skipped(skipped_file, aid, reason)
            time.sleep(args.sleep)
    print("[done] Post-fetch complete.")


# Helper to postfetch a single article URL
def postfetch_one(art_url: str, out: str, max_figs: int, sleep: float, timeout: float, max_retries: int, max_empty_figs: int = 2):
    aid = parse_article_id(art_url)
    print(f"[stream] Fetch: {aid}")
    # verify article page is reachable
    try:
        polite_get(art_url, timeout=timeout, sleep=sleep, max_retries=1)
    except Exception as e:
        print(safe_console(f"[warn] article page not available: {art_url} ({e})"))
        return
    empty_streak = 0
    found_count = 0
    for i in range(1, max_figs + 1):
        fig_url = f"{art_url}/figures/{i}"
        try:
            found = cmd_fig(argparse.Namespace(url=fig_url, out=out, sleep=sleep, timeout=timeout, max_retries=max_retries))
            if not found:
                empty_streak += 1
                if empty_streak >= max_empty_figs:
                    print(f"[info] Stop figures loop for {aid}: consecutive empty pages {empty_streak}")
                    break
            else:
                empty_streak = 0
                found_count += 1
        except Exception as he:
            # stop on repeated 404
            txt = str(he)
            if "404" in txt:
                print(f"[info] Stop figures loop for {aid}: 404 on {i}")
                break
        time.sleep(sleep)
    # return number of figures found; caller decides whether to fetch source data and persist processed record
    return found_count


def postfetch_article(art_url: str, out: str, max_figs: int, max_empty_figs: int, sleep: float, timeout: float, max_retries: int):
    aid = parse_article_id(art_url)
    found = postfetch_one(art_url, out, max_figs, sleep, timeout, max_retries, max_empty_figs)
    if found > 0:
        ok = cmd_source(argparse.Namespace(url=art_url, out=out, section_id=None, filter=None, sleep=sleep, timeout=timeout, max_retries=max_retries))
        if not ok:
            base = Path(out) / aid
            if base.exists():
                shutil.rmtree(base, ignore_errors=True)
            return (aid, -1)
    else:
        base = Path(out) / aid
        if base.exists():
            shutil.rmtree(base, ignore_errors=True)
    return (aid, found)


# ---------- Auto (search multiple -> postfetch) ----------


DEFAULT_KEYWORDS = [
    "cancer", "tumor microenvironment", "immunotherapy", "checkpoint inhibitor",
    "PD-1", "CTLA-4", "CAR-T", "tumour biomarker", "precision oncology",
    "genomics", "transcriptomics", "single-cell", "scRNA-seq", "epigenomics",
    "multi-omics", "CRISPR", "gene editing",
    "machine learning", "deep learning", "foundation model", "graph neural network",
    "metabolism", "metabolomics", "microbiome", "gut microbiota",
    "neuroscience", "brain", "neurodegeneration", "Alzheimer",
    "materials", "quantum", "superconductivity", "perovskite", "catalysis",
    "climate change", "carbon", "ocean", "biodiversity", "ecosystem",
    "bioinformatics", "systems biology", "network biology",
    "COVID-19", "vaccination", "rare disease", "drug discovery",
    "è‚¿ç˜¤å…ç–«", "å…ç–«æ£€æŸ¥ç‚¹", "å•ç»†èƒž", "åŸºå› ç¼–è¾‘", "ä»£è°¢ç»„å­¦", "å¾®ç”Ÿç‰©ç»„",
]


# Expanded built-in keywords (+100)
DEFAULT_KEYWORDS_EXPANDED = [
    # Core oncology/immunology
    "cancer", "tumor microenvironment", "immunotherapy", "checkpoint inhibitor",
    "PD-1", "CTLA-4", "CAR-T", "tumour biomarker", "precision oncology",
    # Omics
    "genomics", "transcriptomics", "single-cell", "scRNA-seq", "epigenomics",
    "multi-omics", "CRISPR", "gene editing",
    # AI/methods
    "machine learning", "deep learning", "foundation model", "graph neural network",
    # Metabolism/microbiome
    "metabolism", "metabolomics", "microbiome", "gut microbiota",
    # Neuroscience
    "neuroscience", "brain", "neurodegeneration", "Alzheimer",
    # Materials/physics/chemistry
    "materials", "quantum", "superconductivity", "perovskite", "catalysis",
    # Climate/earth/environment
    "climate change", "carbon", "ocean", "biodiversity", "ecosystem",
    # Bioinformatics/systems
    "bioinformatics", "systems biology", "network biology",
    # Medicine/public health
    "COVID-19", "vaccination", "rare disease", "drug discovery",
    # Chinese
    "肿瘤免疫", "免疫检查点", "单细胞", "基因编辑", "代谢组学", "微生物组",
    # +100 extended keywords (cross-discipline)
    "proteomics", "lipidomics", "spatial transcriptomics", "spatial omics",
    "single-cell ATAC-seq", "multiome", "perturb-seq", "lineage tracing",
    "organoid", "organoids", "iPSC", "stem cell", "regenerative medicine",
    "CRISPR screen", "base editing", "prime editing", "epigenetic editing",
    "long-read sequencing", "nanopore sequencing", "PacBio", "cryo-EM",
    "X-ray crystallography", "structural biology", "synthetic biology",
    "metabolic engineering", "systems immunology", "metagenomics", "virome",
    "phage therapy", "antibiotic resistance", "exosomes",
    "extracellular vesicles", "liquid biopsy", "circulating tumor DNA",
    "methylation", "ATAC-seq", "ChIP-seq", "Hi-C", "3D genome", "chromatin",
    "enhancer", "super-enhancer", "noncoding RNA", "lncRNA", "microRNA",
    "circRNA", "RNA editing", "m6A", "autophagy", "apoptosis", "ferroptosis",
    "pyroptosis", "cuproptosis", "cellular senescence", "aging", "longevity",
    "mitochondria", "metabolic reprogramming", "immunometabolism",
    "angiogenesis", "metastasis", "epithelial-mesenchymal transition",
    "tumor heterogeneity", "clonal evolution", "phylogenetics",
    "network medicine", "drug repurposing", "AI in medicine",
    "federated learning", "privacy-preserving learning", "causal inference",
    "Mendelian randomization", "GWAS", "fine-mapping", "polygenic risk score",
    "rare variant", "structural variant", "copy number variation", "pangenome",
    "genome assembly", "de novo assembly", "pan-cancer", "AlphaFold",
    "protein design", "deep mutational scanning", "PROTAC", "degrader",
    "antibody-drug conjugate", "nanoparticle delivery", "gene therapy", "AAV",
    "lipid nanoparticles", "mRNA vaccine", "neoantigen", "TCR repertoire",
    "BCR repertoire", "immunopeptidomics", "single-cell multi-omics", "GeoMx",
    "CosMx"
]


def build_default_keywords(min_count: int = 500) -> list[str]:
    base = []
    seen = set()
    def add(term: str):
        t = term.strip()
        if t and t not in seen:
            seen.add(t)
            base.append(t)

    for t in DEFAULT_KEYWORDS_EXPANDED:
        add(t)

    if len(base) >= min_count:
        return base

    diseases = [
        "breast cancer", "lung cancer", "prostate cancer", "colorectal cancer", "melanoma",
        "glioblastoma", "glioma", "leukemia", "lymphoma", "pancreatic cancer",
        "ovarian cancer", "gastric cancer", "liver cancer", "hepatocellular carcinoma",
        "esophageal cancer", "renal cell carcinoma", "endometrial cancer", "sarcoma",
        "multiple myeloma", "head and neck cancer", "pediatric cancer", "rare cancer",
        "metastatic cancer", "brain tumor", "Alzheimer disease", "Parkinson disease",
        "ALS", "multiple sclerosis", "diabetes", "obesity", "NAFLD",
        "cardiovascular disease", "atherosclerosis", "stroke", "hypertension",
        "autoimmune disease", "inflammation", "infection", "COVID-19", "tuberculosis",
        "malaria", "HIV",
    ]
    modalities = [
        "single-cell", "scRNA-seq", "spatial transcriptomics", "multi-omics", "proteomics",
        "metabolomics", "lipidomics", "epigenomics", "ATAC-seq", "ChIP-seq", "Hi-C",
        "CRISPR", "CRISPR screen", "GWAS", "machine learning", "deep learning",
        "foundation model", "graph neural network", "immunotherapy", "checkpoint inhibitor",
        "organoid", "organoids", "gene editing", "base editing", "prime editing",
        "clinical trial", "biomarker", "liquid biopsy",
    ]

    for d in diseases:
        for m in modalities:
            add(f"{d} {m}")
            if len(base) >= min_count:
                return base

    return base
def cmd_auto(args):
    # keywords
    if args.keywords_file:
        kwds = [ln.strip() for ln in Path(args.keywords_file).read_text(encoding="utf-8").splitlines() if ln.strip()]
    else:
        kwds = build_default_keywords(500)
    print(f"[info] Keywords: {len(kwds)} items")

    search_out = Path(args.search_out)
    jsonl_path = search_out / "articles.jsonl"
    # load seen DOIs for dedup when streaming
    seen: set[str] = set()
    if jsonl_path.exists():
        try:
            with jsonl_path.open("r", encoding="utf-8") as f:
                for line in f:
                    try:
                        doi = (json.loads(line).get("doi") or "").lower()
                        if doi:
                            seen.add(doi)
                    except Exception:
                        continue
        except Exception:
            pass

    processed = 0
    # If not streaming, fallback to 2-phase behavior
    if not getattr(args, "stream", False):
        for kw in kwds:
            # Use the same cursor-aware logic as search
            cmd_search(argparse.Namespace(
                query=kw,
                max=args.max_per_keyword,
                out=args.search_out,
                sleep=args.sleep,
                timeout=args.timeout,
                max_retries=args.max_retries,
                mailto=args.mailto,
                no_family_bias=False,
                append=True,
            ))
        # postfetch (always fetch all)
        jsonl = str(jsonl_path)
        
        ns = argparse.Namespace(
            jsonl=jsonl,
            out=args.content_out,
            max_figs=args.max_figs,
            max_articles=args.max_articles,
            sort=args.sort,
            sleep=args.sleep,
            timeout=args.timeout,
            max_retries=args.max_retries,
            workers=getattr(args, "workers", 1),
            processed_file=(args.processed_file or str(Path(args.content_out) / "_processed.txt")),
        )
        cmd_postfetch(ns)
    # Streaming: per-article postfetch immediately after discovery
    processed_file_stream = Path(args.content_out) / "_processed.txt"
    processed_stream = load_processed_set(processed_file_stream)
    skipped_file_stream = processed_file_stream.with_name("_skipped.txt")
    stream_workers = max(1, getattr(args, "stream_workers", 1))
    executor = ThreadPoolExecutor(max_workers=stream_workers, thread_name_prefix="stream-fetch")
    inflight: dict[Any, tuple[int, str]] = {}
    free_slots = list(range(stream_workers))
    stop_stream = False
    total_keywords = len(kwds)
    progress = None
    search_task = None
    fetch_task = None
    worker_tasks: list[int] = []

    def set_worker(slot: int, status: str, detail: str | None = None) -> None:
        if progress is None:
            return
        if slot >= len(worker_tasks):
            return
        msg = f"worker-{slot + 1} {status}"
        if detail:
            clipped = detail[:48]
            msg += f" [{clipped}]"
        progress.update(worker_tasks[slot], description=msg)

    def update_fetch_task() -> None:
        if progress is None or fetch_task is None:
            return
        goal = str(args.max_articles) if args.max_articles else "inf"
        progress.update(fetch_task, description=f"success {processed}/{goal}")

    if console:
        progress = Progress(
            SpinnerColumn(spinner_name="line"),
            TextColumn("[progress.description]{task.description}"),
            TimeElapsedColumn(),
            console=console,
            transient=False,
        )
        progress.start()
        search_task = progress.add_task(f"keywords 0/{total_keywords}", total=None)
        fetch_task = progress.add_task(
            f"success 0/{args.max_articles if args.max_articles else 'inf'}",
            total=None,
        )
        worker_tasks = [
            progress.add_task(f"worker-{idx + 1} idle", total=None)
            for idx in range(stream_workers)
        ]

    def process_futures(blocking: bool) -> None:
        nonlocal processed, stop_stream
        if not inflight:
            return
        futures = list(inflight.keys())
        if blocking:
            done_set, _ = wait(futures, return_when=FIRST_COMPLETED)
        else:
            done_set = [f for f in futures if f.done()]
        for fut in list(done_set):
            slot, aid_hint = inflight.pop(fut)
            free_slots.append(slot)
            try:
                result = fut.result()
            except Exception as exc:
                msg = safe_console(f"[warn] stream fetch failed: {exc}")
                if console:
                    console.log(msg)
                else:
                    print(msg)
                if aid_hint:
                    append_processed(processed_file_stream, aid_hint)
                    append_skipped(skipped_file_stream, aid_hint, "fetch-error")
                    processed_stream.add(aid_hint)
                set_worker(slot, "idle", f"{aid_hint} error")
                continue
            if not result:
                append_processed(processed_file_stream, aid_hint)
                append_skipped(skipped_file_stream, aid_hint, "fetch-error")
                processed_stream.add(aid_hint)
                set_worker(slot, "idle", f"{aid_hint} no-result")
                continue
            aid_out, found = result
            if found > 0:
                if args.max_articles and processed >= args.max_articles:
                    base = Path(args.content_out) / aid_out
                    if base.exists():
                        shutil.rmtree(base, ignore_errors=True)
                    set_worker(slot, "idle", f"{aid_out} drop-limit")
                    stop_stream = True
                else:
                    append_processed(processed_file_stream, aid_out)
                    processed_stream.add(aid_out)
                    processed += 1
                    set_worker(slot, "idle", f"{aid_out} ok")
                    update_fetch_task()
                    if args.max_articles and processed >= args.max_articles:
                        stop_stream = True
            elif found == -1:
                append_processed(processed_file_stream, aid_out)
                append_skipped(skipped_file_stream, aid_out, "no-source-data")
                processed_stream.add(aid_out)
                set_worker(slot, "idle", f"{aid_out} no-source")
            else:
                append_processed(processed_file_stream, aid_out)
                append_skipped(skipped_file_stream, aid_out, "no-figures")
                processed_stream.add(aid_out)
                set_worker(slot, "idle", f"{aid_out} no-image")

    try:
        for idx, kw in enumerate(kwds, 1):
            if stop_stream:
                break
            print(f"[stream] Searching: {safe_console(kw)}")
            if progress is not None and search_task is not None:
                progress.update(search_task, description=f"keywords {idx}/{total_keywords}")
            if args.max_per_keyword > 1000:
                items_iter = crossref_cursor_stream(
                    kw,
                    total_max=args.max_per_keyword,
                    mailto=args.mailto,
                    sleep=args.sleep,
                    timeout=args.timeout,
                    max_retries=args.max_retries,
                    family_bias=True,
                    page_rows=1000,
                )
            else:
                items_iter = crossref_search(
                    kw,
                    rows=args.max_per_keyword,
                    mailto=args.mailto,
                    sleep=args.sleep,
                    timeout=args.timeout,
                    max_retries=args.max_retries,
                    family_bias=True,
                )
            for it in items_iter:
                process_futures(blocking=False)
                if stop_stream:
                    break
                doi = (it.get("DOI") or "").lower()
                if doi and doi in seen:
                    continue
                title_list = it.get("title") or []
                title = title_list[0] if title_list else ""
                container = (it.get("container-title") or [""])[0]
                issued = it.get("issued", {}).get("date-parts", [[None]])[0]
                year = issued[0] if issued else None
                url = it.get("URL")
                abstract = it.get("abstract")
                epmc = europe_pmc_by_doi(doi, sleep=args.sleep, timeout=args.timeout, max_retries=args.max_retries) if doi else None
                pmcid = epmc.get("pmcid") if epmc else None
                abstract_epmc = epmc.get("abstractText") if epmc else None
                is_oa = bool(epmc.get("isOpenAccess") or pmcid) if epmc else False
                pmc_url = f"https://pmc.ncbi.nlm.nih.gov/articles/{pmcid}/" if pmcid else None
                pmid = epmc.get("pmid") if epmc else None
                authors = format_authors_crossref(it.get("author")) or (epmc.get("authorString") if epmc else "")
                rec = {
                    "doi": it.get("DOI"),
                    "title": title,
                    "journal": container,
                    "year": year,
                    "url": url,
                    "pmcid": pmcid,
                    "pmc_url": pmc_url,
                    "pmid": pmid,
                    "is_open_access": is_oa,
                    "abstract": abstract_epmc or abstract or "",
                    "authors": authors,
                }
                append_jsonl(jsonl_path, rec)
                if doi:
                    seen.add(doi)
                art_url = norm_article_url(url, it.get("DOI"))
                if not art_url:
                    continue
                aid2 = parse_article_id(art_url)
                if aid2 in processed_stream:
                    continue
                while not free_slots:
                    process_futures(blocking=True)
                    if stop_stream:
                        break
                if stop_stream:
                    break
                slot = free_slots.pop()
                set_worker(slot, "busy", aid2)
                future = executor.submit(
                    postfetch_article,
                    art_url,
                    args.content_out,
                    args.max_figs,
                    getattr(args, "max_empty_figs", 2),
                    args.sleep,
                    args.timeout,
                    args.max_retries,
                )
                inflight[future] = (slot, aid2)
            if stop_stream:
                break
            process_futures(blocking=False)
            if args.sleep:
                time.sleep(args.sleep)
    finally:
        while inflight:
            process_futures(blocking=True)
        executor.shutdown(wait=True)
        if progress is not None:
            progress.stop()

    print("[done] Streaming search + fetch completed.")


def build_parser():
    p = argparse.ArgumentParser(description="Nature family search + authorized content fetch (all-in-one)")
    sub = p.add_subparsers(dest="cmd", required=True)

    s = sub.add_parser("search", help="Search Nature family via Crossref + Europe PMC")
    s.add_argument("--query", required=True)
    s.add_argument("--max", type=int, default=10)
    s.add_argument("--out", default="outputs/search_run")
    s.add_argument("--mailto", default=None)
    s.add_argument("--sleep", type=float, default=1.0)
    s.add_argument("--timeout", type=float, default=300)
    s.add_argument("--max-retries", type=int, default=3)
    s.add_argument("--append", action="store_true")
    s.add_argument("--no-family-bias", action="store_true")
    s.set_defaults(func=cmd_search)

    f = sub.add_parser("fig", help="Fetch image+caption from a nature.com figure page (authorized)")
    f.add_argument("--url", required=True)
    f.add_argument("--out", default="outputs/nature_content")
    f.add_argument("--sleep", type=float, default=1.0)
    f.add_argument("--timeout", type=float, default=300)
    f.add_argument("--max-retries", type=int, default=3)
    f.set_defaults(func=cmd_fig)

    sd = sub.add_parser("source", help="Fetch Source data from a nature.com article page (authorized)")
    sd.add_argument("--url", required=True)
    sd.add_argument("--out", default="outputs/nature_content")
    sd.add_argument("--section-id", default=None)
    sd.add_argument("--filter", default=None)
    sd.add_argument("--sleep", type=float, default=1.0)
    sd.add_argument("--timeout", type=float, default=300)
    sd.add_argument("--max-retries", type=int, default=3)
    sd.set_defaults(func=cmd_source)

    pf = sub.add_parser("postfetch", help="Fetch ALL (figures + source data) for articles in JSONL")
    pf.add_argument("--jsonl", required=True)
    pf.add_argument("--out", default="outputs/nature_content")
    pf.add_argument("--max-figs", type=int, default=12)
    pf.add_argument("--max-articles", type=int, default=0)
    pf.add_argument("--max-empty-figs", type=int, default=2, help="Max consecutive empty figure pages before stopping")
    pf.add_argument("--sort", choices=["year_desc", "year_asc", "input"], default="year_desc")
    pf.add_argument("--sleep", type=float, default=1.0)
    pf.add_argument("--timeout", type=float, default=300)
    pf.add_argument("--max-retries", type=int, default=3)
    # Accept but ignore --max-per-keyword for compatibility with auto
    pf.add_argument("--max-per-keyword", type=int, default=None, help="(compat) accepted but ignored in postfetch; used in auto search")
    pf.add_argument("--workers", type=int, default=1, help="Worker processes for fetching (non-stream mode)")
    pf.add_argument("--processed-file", default=None, help="Path to processed record file (default: <out>/_processed.txt)")
    pf.set_defaults(func=cmd_postfetch)

    au = sub.add_parser("auto", help="Search multiple keywords then fetch ALL content")
    au.add_argument("--keywords-file", default=None)
    au.add_argument("--max-per-keyword", type=int, default=50)
    au.add_argument("--search-out", default="outputs/search_auto")
    au.add_argument("--content-out", default="outputs/nature_content")
    au.add_argument("--mailto", default=None)
    au.add_argument("--sleep", type=float, default=1.0)
    au.add_argument("--timeout", type=float, default=300)
    au.add_argument("--max-retries", type=int, default=3)
    au.add_argument("--max-articles", type=int, default=0)
    au.add_argument("--max-figs", type=int, default=12)
    au.add_argument("--max-empty-figs", type=int, default=2)
    au.add_argument("--sort", choices=["year_desc", "year_asc", "input"], default="year_desc")
    au.add_argument("--workers", type=int, default=1, help="Workers for postfetch in non-stream mode (processes)")
    au.add_argument("--processed-file", default=None, help="Path to processed record file (default: <content-out>/_processed.txt)")
    au.add_argument("--stream", action="store_true", help="Enable streaming mode: per-article immediate fetch (no need to wait for all searches)")
    au.add_argument("--stream-workers", type=int, default=1, help="Streaming fetch worker threads (>=1)")
    au.set_defaults(func=cmd_auto)

    return p


def main():
    parser = build_parser()
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()





