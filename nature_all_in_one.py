﻿#!/usr/bin/env python3
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
import os
import re
import subprocess
import sys
import time
from pathlib import Path
from typing import Any
from urllib.parse import urljoin, urlparse, unquote


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
    items = crossref_search(args.query, rows=args.max, mailto=args.mailto, sleep=args.sleep, timeout=args.timeout, max_retries=args.max_retries, family_bias=not args.no_family_bias)
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


def pick_largest_src(soup: BeautifulSoup, base_url: str):
    og = soup.find("meta", attrs={"property": "og:image"})
    if og and og.get("content"):
        cand = urljoin(base_url, og["content"].strip())
        if is_likely_image_url(cand):
            return cand
    fig = soup.find("figure")
    if fig:
        src = fig.find("source")
        if src and src.get("srcset"):
            entries = [x.strip() for x in src["srcset"].split(",") if x.strip()]
            for entry in reversed(entries):
                u = urljoin(base_url, entry.split()[0])
                if is_likely_image_url(u):
                    return u
        img = fig.find("img")
    else:
        img = soup.find("img")
    if img and img.get("src"):
        cand = urljoin(base_url, img["src"].strip())
        if is_likely_image_url(cand):
            return cand
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


def download_binary(url: str, out_path: Path, referer: str | None = None, timeout=30, sleep=1.0, max_retries=3):
    headers = {"User-Agent": API_USER_AGENT, "Accept": "*/*", "Connection": "close"}
    if referer:
        headers["Referer"] = referer
    r = polite_get(url, timeout=timeout, sleep=sleep, max_retries=max_retries, headers=headers)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("wb") as f:
        f.write(r.content)
    return str(out_path)


def parse_article_id_and_fig(url: str):
    m = re.search(r"/articles/([^/]+)(?:/figures/(\d+))?", url)
    if not m:
        return ("unknown", None)
    return (m.group(1), int(m.group(2)) if m.group(2) else None)


def cmd_fig(args):
    url = args.url
    aid, fno = parse_article_id_and_fig(url)
    print(f"[info] Article: {aid} | Figure: {fno if fno else 'all/unknown'}")
    r = polite_get(url, timeout=args.timeout, sleep=args.sleep, max_retries=args.max_retries)
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

    saved_cap = None
    if caption:
        out_cap = figures_dir / f"{fig_tag}.txt"
        out_cap.write_text(caption, encoding="utf-8")
        saved_cap = str(out_cap)
        print("[done] Caption saved.")
    else:
        print("[warn] No caption text found.")

    entry = {"figure_tag": fig_tag, "figure_no": fno, "image_file": saved_img, "caption_file": saved_cap, "image_url": img_url, "source_url": url}
    upsert_json_list(meta_dir / "figures.json", entry, key="figure_tag")


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
        print(f"  [{i}/{len(links)}] {safe_console(label)} -> {fname}")
        try:
            saved = download_binary(file_url, save_to, referer=r.url, timeout=args.timeout, sleep=args.sleep, max_retries=args.max_retries)
            entry = {"label": label, "url": file_url, "saved_as": str(save_to), "orig_name": fname_url}
            manifest.append(entry)
            upsert_json_list(meta_dir / "source_data.json", entry, key="label")
        except Exception as e:
            entry = {"label": label, "url": file_url, "error": str(e), "orig_name": fname_url}
            manifest.append(entry)
            upsert_json_list(meta_dir / "source_data.json", entry, key="label")
    (meta_dir / "_source_data_manifest.json").write_text(json.dumps({"article_url": url, "links": manifest}, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[done] Saved manifest and files under {base}")


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

    taken = 0
    for idx, r in enumerate(rows, 1):
        art_url = norm_article_url(r.get("url"), r.get("doi"))
        if not art_url:
            continue
        aid = parse_article_id(art_url)
        print(f"[{idx}] Nature article detected: {aid}")
        for i in range(1, args.max_figs + 1):
            fig_url = f"{art_url}/figures/{i}"
            cmd_fig(argparse.Namespace(url=fig_url, out=args.out, sleep=args.sleep, timeout=args.timeout, max_retries=args.max_retries))
            time.sleep(args.sleep)
        cmd_source(argparse.Namespace(url=art_url, out=args.out, section_id=None, filter=None, sleep=args.sleep, timeout=args.timeout, max_retries=args.max_retries))
        time.sleep(args.sleep)
        taken += 1
        if args.max_articles and taken >= args.max_articles:
            break
    print("[done] Post-fetch complete.")


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
]def cmd_auto(args):
    # keywords
    if args.keywords_file:
        kwds = [ln.strip() for ln in Path(args.keywords_file).read_text(encoding="utf-8").splitlines() if ln.strip()]
    else:
        kwds = DEFAULT_KEYWORDS_EXPANDED
    print(f"[info] Keywords: {len(kwds)} items")
    # search (append + dedup)
    for kw in kwds:
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
    jsonl = str(Path(args.search_out) / "articles.jsonl")
    cmd_postfetch(argparse.Namespace(
        jsonl=jsonl,
        out=args.content_out,
        max_figs=args.max_figs,
        max_articles=args.max_articles,
        sort=args.sort,
        sleep=args.sleep,
        timeout=args.timeout,
        max_retries=args.max_retries,
    ))
    print("[done] Search and full-content fetch completed.")


def build_parser():
    p = argparse.ArgumentParser(description="Nature family search + authorized content fetch (all-in-one)")
    sub = p.add_subparsers(dest="cmd", required=True)

    s = sub.add_parser("search", help="Search Nature family via Crossref + Europe PMC")
    s.add_argument("--query", required=True)
    s.add_argument("--max", type=int, default=10)
    s.add_argument("--out", default="outputs/search_run")
    s.add_argument("--mailto", default=None)
    s.add_argument("--sleep", type=float, default=1.0)
    s.add_argument("--timeout", type=float, default=30)
    s.add_argument("--max-retries", type=int, default=3)
    s.add_argument("--append", action="store_true")
    s.add_argument("--no-family-bias", action="store_true")
    s.set_defaults(func=cmd_search)

    f = sub.add_parser("fig", help="Fetch image+caption from a nature.com figure page (authorized)")
    f.add_argument("--url", required=True)
    f.add_argument("--out", default="outputs/nature_content")
    f.add_argument("--sleep", type=float, default=1.0)
    f.add_argument("--timeout", type=float, default=30)
    f.add_argument("--max-retries", type=int, default=3)
    f.set_defaults(func=cmd_fig)

    sd = sub.add_parser("source", help="Fetch Source data from a nature.com article page (authorized)")
    sd.add_argument("--url", required=True)
    sd.add_argument("--out", default="outputs/nature_content")
    sd.add_argument("--section-id", default=None)
    sd.add_argument("--filter", default=None)
    sd.add_argument("--sleep", type=float, default=1.0)
    sd.add_argument("--timeout", type=float, default=30)
    sd.add_argument("--max-retries", type=int, default=3)
    sd.set_defaults(func=cmd_source)

    pf = sub.add_parser("postfetch", help="Fetch ALL (figures + source data) for articles in JSONL")
    pf.add_argument("--jsonl", required=True)
    pf.add_argument("--out", default="outputs/nature_content")
    pf.add_argument("--max-figs", type=int, default=12)
    pf.add_argument("--max-articles", type=int, default=0)
    pf.add_argument("--sort", choices=["year_desc", "year_asc", "input"], default="year_desc")
    pf.add_argument("--sleep", type=float, default=1.0)
    pf.add_argument("--timeout", type=float, default=30)
    pf.add_argument("--max-retries", type=int, default=3)
    pf.set_defaults(func=cmd_postfetch)

    au = sub.add_parser("auto", help="Search multiple keywords then fetch ALL content")
    au.add_argument("--keywords-file", default=None)
    au.add_argument("--max-per-keyword", type=int, default=50)
    au.add_argument("--search-out", default="outputs/search_auto")
    au.add_argument("--content-out", default="outputs/nature_content")
    au.add_argument("--mailto", default=None)
    au.add_argument("--sleep", type=float, default=1.0)
    au.add_argument("--timeout", type=float, default=30)
    au.add_argument("--max-retries", type=int, default=3)
    au.add_argument("--max-articles", type=int, default=0)
    au.add_argument("--max-figs", type=int, default=12)
    au.add_argument("--sort", choices=["year_desc", "year_asc", "input"], default="year_desc")
    au.set_defaults(func=cmd_auto)

    return p


def main():
    parser = build_parser()
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()

