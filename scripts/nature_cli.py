#!/usr/bin/env python3
import argparse
import csv
import importlib
import json
import os
import re
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


API_USER_AGENT = (
    "PheroViz-NatureMeta/0.2 (+compliant; no-scrape; contact=local)"
)


def safe_console(text: str) -> str:
    try:
        enc = sys.stdout.encoding or "utf-8"
        return text.encode(enc, errors="replace").decode(enc, errors="replace")
    except Exception:
        # best-effort fallback
        try:
            return text.encode("ascii", errors="replace").decode("ascii")
        except Exception:
            return "<unprintable>"


def is_nature_family(container_titles) -> bool:
    if not container_titles:
        return False
    if isinstance(container_titles, str):
        titles = [container_titles]
    else:
        titles = container_titles
    for t in titles:
        if not t:
            continue
        tt = t.strip()
        if tt == "Nature" or tt.lower().startswith("nature "):
            return True
    return False


def polite_get(url: str, params=None, timeout=30, sleep=1.0, max_retries=3):
    headers = {"User-Agent": API_USER_AGENT}
    attempt = 0
    backoff = sleep if sleep else 0.5
    while True:
        attempt += 1
        try:
            resp = requests.get(url, params=params, headers=headers, timeout=timeout)
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


def crossref_search(query: str, rows: int = 20, mailto: str | None = None, sleep=1.0, timeout=30, max_retries=3, family_bias=True):
    base = "https://api.crossref.org/works"
    params = {
        "query": query,
        # narrow to journal articles; we'll filter Nature-family after fetch
        "filter": "type:journal-article",
        "rows": rows,
        # Crossref recommends mailto, but we'll omit if not provided
    }
    if family_bias:
        params["query.container-title"] = "Nature"
    if mailto:
        params["mailto"] = mailto
    r = polite_get(base, params=params, sleep=sleep, timeout=timeout, max_retries=max_retries)
    data = r.json()
    items = data.get("message", {}).get("items", [])
    # filter to Nature-family titles
    filtered = [it for it in items if is_nature_family(it.get("container-title"))]
    return filtered


def europe_pmc_by_doi(doi: str, sleep=1.0, timeout=30, max_retries=3):
    url = "https://www.ebi.ac.uk/europepmc/webservices/rest/search"
    params = {"query": f"DOI:{doi}", "format": "json", "pageSize": 1}
    r = polite_get(url, params=params, sleep=sleep, timeout=timeout, max_retries=max_retries)
    data = r.json()
    results = data.get("resultList", {}).get("result", [])
    if not results:
        return None
    return results[0]


def fetch_pmc_figure_urls(pmcid: str, sleep=1.0, timeout=30, max_retries=3):
    # Parse the PMC article HTML and collect figure image URLs.
    # Only used for OA PMC records.
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
    # dedupe while preserving order
    seen = set()
    unique = []
    for u in urls:
        if u not in seen:
            seen.add(u)
            unique.append(u)
    return unique


def sanitize_filename(s: str) -> str:
    return re.sub(r"[^a-zA-Z0-9_.-]+", "_", s)


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


def download_image(url: str, outdir: Path, idx: int, sleep=1.0, timeout=30, max_retries=3):
    try:
        r = polite_get(url, sleep=sleep, timeout=timeout, max_retries=max_retries)
    except Exception as e:
        print(f"  [warn] image fetch failed: {url} ({e})")
        return None
    ctype = r.headers.get("Content-Type", "").lower()
    ext = ".jpg"
    if "png" in ctype:
        ext = ".png"
    elif "gif" in ctype:
        ext = ".gif"
    elif "jpeg" in ctype or "jpg" in ctype:
        ext = ".jpg"
    else:
        # try from URL
        m = re.search(r"\.(png|jpg|jpeg|gif)(?:\?|$)", url, re.I)
        if m:
            ext = "." + m.group(1).lower().replace("jpeg", "jpg")
    outdir.mkdir(parents=True, exist_ok=True)
    out = outdir / f"fig_{idx:03d}{ext}"
    with out.open("wb") as f:
        f.write(r.content)
    return str(out)


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
    # load existing
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
    # add new
    for r in records:
        doi = (r.get("doi") or "").lower()
        if doi and doi not in seen:
            seen.add(doi)
            merged.append(r)
    return merged


def main():
    p = argparse.ArgumentParser(description="Compliant Nature family search via Crossref + Europe PMC")
    p.add_argument("--query", required=True, help="Search query (e.g., keywords)")
    p.add_argument("--max", type=int, default=10, help="Max results to retrieve (upper bound)")
    p.add_argument("--out", default="outputs/run", help="Output directory")
    p.add_argument("--mailto", default=None, help="Optional email for Crossref mailto")
    p.add_argument("--images", action="store_true", help="Download PMC figure images when available")
    p.add_argument("--sleep", type=float, default=1.0, help="Seconds to sleep between requests")
    p.add_argument("--timeout", type=float, default=30, help="Request timeout seconds")
    p.add_argument("--max-retries", type=int, default=3, help="Max retries per request")
    p.add_argument("--append", action="store_true", help="Append to existing JSONL with DOI de-duplication")
    p.add_argument("--no-family-bias", action="store_true", help="Do not bias Crossref query by container-title=Nature")
    args = p.parse_args()

    outdir = Path(args.out)
    outdir.mkdir(parents=True, exist_ok=True)

    print(f"[info] Query: {safe_console(args.query)}")
    print(f"[info] Max: {args.max} | Images: {args.images} | Sleep: {args.sleep}s")
    
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

    records: list[dict] = []
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
        pmcid = None
        abstract_epmc = None
        is_oa = False
        pmc_url = None
        pmid = None
        author_str = None
        if epmc:
            pmcid = epmc.get("pmcid")
            abstract_epmc = epmc.get("abstractText")
            is_oa = bool(epmc.get("isOpenAccess") or pmcid)
            if pmcid:
                pmc_url = f"https://pmc.ncbi.nlm.nih.gov/articles/{pmcid}/"
            pmid = epmc.get("pmid")
            author_str = epmc.get("authorString")

        final_abstract = abstract_epmc or abstract or ""
        # authors from Crossref preferred; fallback to Europe PMC authorString
        authors_crossref = format_authors_crossref(it.get("author"))
        authors = authors_crossref or author_str or ""
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

        ttl = safe_console(title)
        jn = safe_console(container)
        print(f"[{i}/{len(items)}] {ttl[:80]}...")
        print(f"      DOI: {doi} | Journal: {jn} | Year: {year}")
        if pmcid:
            print(f"      PMC: {pmcid} (OA: {is_oa})")

        # optional image download for PMC OA
        image_paths: list[str] = []
        if args.images and pmcid:
            try:
                urls = fetch_pmc_figure_urls(pmcid, sleep=args.sleep, timeout=args.timeout, max_retries=args.max_retries)
                print(f"      Figures found: {len(urls)}")
                figdir = outdir / "images" / sanitize_filename(doi or title)
                for idx, u in enumerate(urls, 1):
                    saved = download_image(u, figdir, idx, sleep=args.sleep, timeout=args.timeout, max_retries=args.max_retries)
                    if saved:
                        image_paths.append(saved)
            except Exception as e:
                print(f"      [warn] failed to fetch images for {pmcid}: {e}")

        rec["figure_images"] = image_paths
        records.append(rec)

    # write outputs
    jsonl_path = outdir / "articles.jsonl"
    if args.append and jsonl_path.exists():
        merged = merge_append(records, jsonl_path)
        write_jsonl(jsonl_path, merged)
    else:
        write_jsonl(jsonl_path, records)
    fields = [
        "doi",
        "title",
        "journal",
        "year",
        "url",
        "pmcid",
        "pmc_url",
        "pmid",
        "is_open_access",
        "abstract",
        "authors",
    ]
    write_csv(outdir / "articles.csv", records, fields)
    print(f"[done] Saved {len(records)} records to {outdir}")


if __name__ == "__main__":
    main()
