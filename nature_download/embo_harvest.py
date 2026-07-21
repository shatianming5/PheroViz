#!/usr/bin/env python3
"""Harvest CC-BY EMBO Press articles on SpringerLink with source data.

Figure-to-source association is attempted strictly first: for each SpringerLink
``Fig<N>`` caption block, the parser inspects the figure container plus nearby
following siblings for a ``Source data ... for this figure`` link whose href
contains ``#MOESM<k>``, then resolves that key through the article's static
Springer supplementary-file map. If an article exposes per-figure source links
but the local HTML nesting prevents a reliable per-figure association, the
parser falls back to admitting an otherwise eligible article and attaches all
resolvable per-figure source files found in the article.
"""

from __future__ import annotations

import argparse
from datetime import date, datetime, timezone
import html
import json
import os
from pathlib import Path
import re
import shutil
import socket
import time
from typing import Any, Iterator
from urllib.parse import quote, unquote, urlparse
import zipfile

from bs4 import BeautifulSoup, Tag
import requests

from corpus.completeness import (
    inspect_article_payload,
    quarantine_article_path,
    quarantine_incomplete_article,
    require_complete_article_payload,
)
from corpus.policy import evaluate_crossref_item
from corpus.provenance import build_article_manifest


CROSSREF_URL = "https://api.crossref.org/works"
CROSSREF_MAILTO = "nature-vis-corpus@outlook.com"
DEFAULT_FROM_DATE = "2024-01-01"
DEFAULT_UNTIL_DATE = date.today().isoformat()
USER_AGENT = (
    "PheroViz-EMBO-harvester/1.0 "
    "(CC-BY research corpus; https://github.com/shatianming5/PheroViz; "
    "mailto:nature-vis-corpus@outlook.com)"
)
API_USER_AGENT = USER_AGENT

EMBO_JOURNALS: dict[str, dict[str, str]] = {
    "1460-2075": {"name": "EMBO Journal", "prefix": "10.1038/s44318"},
    "1744-4292": {"name": "Molecular Systems Biology", "prefix": "10.1038/s44320"},
    "1469-3178": {"name": "EMBO Reports", "prefix": "10.1038/s44319"},
    "1757-4684": {"name": "EMBO Molecular Medicine", "prefix": "10.1038/s44321"},
}
JOURNAL_TO_ISSN = {
    info["name"].casefold(): issn for issn, info in EMBO_JOURNALS.items()
}
CC_BY_4_HTML_PATTERN = re.compile(r"creativecommons\.org/licenses/by/4", re.I)
CC_BY_4_PATTERN = re.compile(
    r"^https?://(?:www\.)?creativecommons\.org/licenses/by/4\.0(?:/|$)",
    re.I,
)
DOI_PATTERN = re.compile(
    r"^(?:10\.1038/s443(?:18|19|20|21)-\d{3}-\d{5}-\d"
    r"|10\.15252/(?:embj|embr|msb|emmm)\.\S+)$",
    re.I,
)
SPRINGER_SOURCE_PATTERN = re.compile(
    r"(?:https:)?//static-content\.springer\.com/esm/art%3A(?P<doi>[^/\s\"'<>]+)"
    r"/MediaObjects/(?P<name>[^\s\"'<>?#]+_MOESM(?P<moesm>\d+)_ESM\."
    r"(?P<ext>xlsx|csv|zip))(?:[?#][^\s\"'<>]*)?",
    re.I,
)


def is_archive_junk(member: str) -> bool:
    """True for OS/app-generated junk zip members that are never real data:
    macOS ``__MACOSX/`` resource-fork dirs, ``._``-prefixed AppleDouble shadow
    files, and ``~$``-prefixed Microsoft Office lock/owner temp files."""
    name = Path(member).name
    return (
        "__MACOSX" in Path(member).parts
        or name.startswith("._")
        or name.startswith("~$")
    )
FIG_IMAGE_PATTERN = re.compile(
    r"(?:https:)?//media\.springernature\.com/(?P<size>lw\d+|full)/"
    r"springer-static/image/art%3A(?P<doi>[^/\s\"'<>]+)/MediaObjects/"
    r"(?P<name>[^\s\"'<>?#]+_Fig(?P<fig>\d+)_HTML\.png)(?:[?#][^\s\"'<>]*)?",
    re.I,
)
PANEL_GROUP_PATTERN = re.compile(
    r"\(\s*([A-L](?:\s*[,;/&+\-–—]\s*[A-L])*)\s*\)",
    re.I,
)
PANEL_SINGLE_PATTERN = re.compile(r"\(\s*([A-L])\s*\)", re.I)
BOLD_PANEL_HTML_PATTERN = re.compile(r"<(?:b|strong)\b[^>]*>\s*([A-L])\s*</(?:b|strong)>\s*[,.:;)]", re.I)
TEXT_PANEL_PATTERN = re.compile(r"(?:^|[\s.;:])([A-L])\s*[,.:;]\s+", re.I)
SOURCE_LINK_TEXT_PATTERN = re.compile(r"source\s+data.*for\s+this\s+figure", re.I | re.S)


def utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def load_processed(path: Path) -> set[str]:
    if not path.is_file():
        return set()
    return {line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()}


def append_line(path: Path, line: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(line + "\n")


def request_with_retries(
    session: requests.Session,
    url: str,
    *,
    params: dict[str, Any] | None = None,
    referer: str | None = None,
    accept: str | None = None,
    timeout: float,
    sleep: float,
    max_retries: int,
    stream: bool = False,
) -> requests.Response:
    headers: dict[str, str] = {}
    if referer:
        headers["Referer"] = referer
    if accept:
        headers["Accept"] = accept
    req_timeout: tuple[float, float] | float = (min(6.0, timeout), timeout)
    for attempt in range(1, max_retries + 1):
        try:
            response = session.get(
                url,
                params=params,
                headers=headers,
                timeout=req_timeout,
                stream=stream,
                allow_redirects=True,
            )
            response.raise_for_status()
            if sleep:
                time.sleep(sleep)
            return response
        except requests.RequestException as exc:
            status = getattr(getattr(exc, "response", None), "status_code", None)
            if isinstance(exc, (requests.Timeout, requests.ConnectionError)):
                try:
                    session.close()
                except Exception:
                    pass
            if attempt == max_retries:
                raise
            wait = min(60.0, 8.0 * (2 ** (attempt - 1))) if status in (429, 403) else max(sleep, 0.5) * (2 ** (attempt - 1))
            print(
                f"  [retry {attempt}/{max_retries}] {url}: {type(exc).__name__}"
                f"{f' {status}' if status else ''}; waiting {wait:.1f}s"
            )
            time.sleep(wait)
    raise RuntimeError("request retry loop exhausted")


def is_cc_by_4(item: dict[str, Any]) -> bool:
    return any(
        CC_BY_4_PATTERN.search(str(record.get("URL") or record.get("url") or ""))
        for record in item.get("license") or []
        if isinstance(record, dict)
    )


def article_id_from_doi(doi: str) -> str | None:
    doi = doi.strip().casefold()
    if not DOI_PATTERN.fullmatch(doi):
        return None
    suffix = doi.split("/", 1)[1]
    return re.sub(r"[^a-z0-9._-]+", "-", suffix).strip("-")


def article_url(doi: str) -> str:
    return f"https://link.springer.com/article/{doi}"


def panel_count(caption: str) -> int:
    letters: set[str] = set()
    for match in PANEL_GROUP_PATTERN.finditer(caption):
        letters.update(re.findall(r"[A-L]", match.group(1), flags=re.I))
    letters.update(match.group(1).upper() for match in PANEL_SINGLE_PATTERN.finditer(caption))
    letters.update(match.group(1).upper() for match in BOLD_PANEL_HTML_PATTERN.finditer(caption))
    text = BeautifulSoup(caption, "html.parser").get_text(" ") if "<" in caption else caption
    letters.update(match.group(1).upper() for match in TEXT_PANEL_PATTERN.finditer(text))
    return max((ord(letter.upper()) - ord("A") + 1 for letter in letters), default=0)


def _dedupe(values: Iterator[str]) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []
    for value in values:
        if value not in seen:
            seen.add(value)
            out.append(value)
    return out


def _absolute_url(url: str) -> str:
    return f"https:{url}" if url.startswith("//") else url


def build_moesm_source_map(html_text: str, doi: str) -> dict[str, dict[str, str]]:
    decoded = html.unescape(html_text)
    encoded_doi = quote(doi.casefold(), safe="").casefold()
    source_map: dict[str, dict[str, str]] = {}
    for match in SPRINGER_SOURCE_PATTERN.finditer(decoded):
        if match.group("doi").casefold() != encoded_doi:
            continue
        key = f"MOESM{int(match.group('moesm'))}"
        url = _absolute_url(match.group(0))
        source_map[key] = {
            "url": url,
            "name": match.group("name"),
            "ext": match.group("ext").casefold(),
        }
    return source_map


def figure_image_urls(html_text: str, doi: str) -> dict[int, str]:
    decoded = html.unescape(html_text)
    encoded_doi = quote(doi.casefold(), safe="").casefold()
    best: dict[int, tuple[int, str]] = {}
    for match in FIG_IMAGE_PATTERN.finditer(decoded):
        if match.group("doi").casefold() != encoded_doi:
            continue
        fig_no = int(match.group("fig"))
        size = match.group("size").casefold()
        width = 99999 if size == "full" else int(size.removeprefix("lw"))
        url = _absolute_url(match.group(0))
        if fig_no not in best or width > best[fig_no][0]:
            best[fig_no] = (width, url)
    return {fig_no: url for fig_no, (_, url) in best.items()}


def has_cc_by_4_license(html_text: str) -> bool:
    return bool(CC_BY_4_HTML_PATTERN.search(html_text))


def _snippet_for_tag(tag: Tag, sibling_count: int = 5) -> str:
    figure_parent = tag.find_parent("figure")
    if figure_parent is not None:
        return str(figure_parent)
    parts = [str(tag)]
    parent = tag.parent if isinstance(tag.parent, Tag) else None
    if parent is not None:
        parts.insert(0, str(parent))
        sibling = parent.next_sibling
    else:
        sibling = tag.next_sibling
    seen = 0
    while sibling is not None and seen < sibling_count:
        if isinstance(sibling, Tag):
            parts.append(str(sibling))
            seen += 1
        sibling = sibling.next_sibling
    return "\n".join(parts)


def _moesm_keys_in_per_figure_source_links(snippet: str) -> list[str]:
    soup = BeautifulSoup(snippet, "html.parser")
    keys: list[str] = []
    for anchor in soup.find_all("a", href=True):
        href = str(anchor.get("href") or "")
        text = anchor.get_text(" ", strip=True)
        combined = f"{text} {href}"
        if not SOURCE_LINK_TEXT_PATTERN.search(combined):
            continue
        for raw in re.findall(r"#MOESM(\d+)", href, flags=re.I):
            keys.append(f"MOESM{int(raw)}")
    if not keys and SOURCE_LINK_TEXT_PATTERN.search(soup.get_text(" ", strip=True)):
        for raw in re.findall(r"#MOESM(\d+)", snippet, flags=re.I):
            keys.append(f"MOESM{int(raw)}")
    return _dedupe(iter(keys))


def all_per_figure_source_keys(html_text: str) -> list[str]:
    return _moesm_keys_in_per_figure_source_links(html_text)


def parse_article_assets(html_text: str, doi: str) -> tuple[list[dict[str, Any]], list[str]]:
    soup = BeautifulSoup(html_text, "html.parser")
    source_map = build_moesm_source_map(html_text, doi)
    image_by_fig = figure_image_urls(html_text, doi)
    per_figure_keys = all_per_figure_source_keys(html_text)
    figures: list[dict[str, Any]] = []
    fallback_used = False
    for caption_tag in soup.find_all(id=re.compile(r"^Fig\d+$", re.I)):
        match = re.fullmatch(r"Fig(\d+)", str(caption_tag.get("id") or ""), flags=re.I)
        if not match:
            continue
        number = int(match.group(1))
        caption_html = str(caption_tag)
        caption_text = caption_tag.get_text(" ", strip=True)
        snippet = _snippet_for_tag(caption_tag)
        figure_text = BeautifulSoup(snippet, "html.parser").get_text(" ", strip=True)
        snippet_images = figure_image_urls(snippet, doi)
        keys = [key for key in _moesm_keys_in_per_figure_source_links(snippet) if key in source_map]
        if not keys and per_figure_keys:
            keys = [key for key in per_figure_keys if key in source_map]
            fallback_used = bool(keys)
        sources = [{**source_map[key], "moesm_key": key} for key in keys]
        figures.append(
            {
                "number": number,
                "caption": figure_text or caption_text,
                "panels": panel_count(f"{figure_text} {snippet} {caption_html}"),
                "image_url": image_by_fig.get(number) or next(iter(snippet_images.values()), None),
                "source_records": sources,
                "source_keys": keys,
                "source_association": "article-per-figure-fallback" if fallback_used and keys else "caption-neighborhood",
            }
        )
    return sorted(figures, key=lambda figure: figure["number"]), per_figure_keys


def download_file(
    session: requests.Session,
    url: str,
    path: Path,
    *,
    referer: str,
    timeout: float,
    sleep: float,
    max_retries: int,
) -> str:
    response = request_with_retries(
        session,
        url,
        referer=referer,
        timeout=timeout,
        sleep=sleep,
        max_retries=max_retries,
        stream=True,
    )
    part = path.with_name(f".{path.name}.part")
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        with part.open("wb") as handle:
            for chunk in response.iter_content(1024 * 1024):
                if chunk:
                    handle.write(chunk)
        if not part.stat().st_size:
            raise ValueError(f"empty download: {url}")
        part.replace(path)
    finally:
        response.close()
        if part.exists():
            part.unlink()
    return response.headers.get("Content-Type", "").split(";", 1)[0]


def fetch_article_html(
    session: requests.Session,
    doi: str,
    *,
    timeout: float,
    sleep: float,
    max_retries: int,
) -> str:
    response = request_with_retries(
        session,
        article_url(doi),
        accept="text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
        timeout=timeout,
        sleep=sleep,
        max_retries=max_retries,
    )
    return response.text


def harvest_article(
    session: requests.Session,
    item: dict[str, Any],
    *,
    out: Path,
    min_panels: int,
    timeout: float,
    sleep: float,
    max_retries: int,
) -> tuple[bool, str]:
    doi = str(item.get("DOI") or "").casefold()
    article_id = article_id_from_doi(doi)
    if article_id is None:
        return False, "invalid-doi"
    url = article_url(doi)
    html_text = fetch_article_html(session, doi, timeout=timeout, sleep=sleep, max_retries=max_retries)
    if not has_cc_by_4_license(html_text):
        return False, "html-license-not-cc-by-4"
    figures, _ = parse_article_assets(html_text, doi)
    if not figures:
        return False, "no-figures"
    eligible_figures = [
        figure for figure in figures
        if figure["panels"] >= min_panels
        and figure.get("image_url")
        and any(source["ext"] in {"xlsx", "csv", "zip"} for source in figure.get("source_records") or [])
    ]
    if not eligible_figures:
        return False, "no-eligible-figure-source-data"
    selected = max(eligible_figures, key=lambda figure: (figure["panels"], -figure["number"]))

    base = out / article_id
    staging = out / f".{article_id}.partial"
    shutil.rmtree(staging, ignore_errors=True)
    source_dir = staging / "source_data"
    figure_dir = staging / "figures"
    meta_dir = staging / "meta"
    source_dir.mkdir(parents=True)
    figure_dir.mkdir()
    meta_dir.mkdir()
    source_records: list[dict[str, Any]] = []
    figure_records: list[dict[str, Any]] = []
    try:
        source_urls: list[dict[str, str]] = []
        seen_urls: set[str] = set()
        for source in selected["source_records"]:
            if source["ext"] not in {"xlsx", "csv", "zip"} or source["url"] in seen_urls:
                continue
            seen_urls.add(source["url"])
            source_urls.append(source)
        def add_source_record(
            *,
            source: dict[str, str],
            saved: Path,
            content_type: str,
            orig_name: str,
            content_name: str | None,
        ) -> None:
            source_records.append(
                {
                    "label": f"EMBO source data ({saved.name})",
                    "url": source["url"],
                    "saved_as": str(out / article_id / "source_data" / saved.name),
                    "saved_name": saved.name,
                    "orig_name": orig_name,
                    "content_name": content_name,
                    "content_type": content_type,
                    "moesm_key": source.get("moesm_key"),
                    "association": selected.get("source_association"),
                }
            )

        for source in source_urls:
            name = Path(unquote(urlparse(source["url"]).path)).name
            saved = source_dir / name
            content_type = download_file(
                session,
                source["url"],
                saved,
                referer=url,
                timeout=timeout,
                sleep=sleep,
                max_retries=max_retries,
            )
            if source["ext"] in {"xlsx", "zip"} and not zipfile.is_zipfile(saved):
                raise ValueError(f"download is not a valid {source['ext'].upper()} archive: {source['url']}")
            if source["ext"] != "zip":
                add_source_record(
                    source=source,
                    saved=saved,
                    content_type=content_type,
                    orig_name=name,
                    content_name=None,
                )
                continue
            extracted = 0
            with zipfile.ZipFile(saved) as archive:
                for member in archive.namelist():
                    member_name = Path(member).name
                    if is_archive_junk(member):
                        continue
                    if not member_name or Path(member_name).suffix.casefold() not in {".xlsx", ".csv"}:
                        continue
                    target = source_dir / f"{Path(name).stem}_{member_name}"
                    target.write_bytes(archive.read(member))
                    if target.suffix.casefold() == ".xlsx" and not zipfile.is_zipfile(target):
                        raise ValueError(f"zip member is not a valid XLSX archive: {member}")
                    extracted += 1
                    add_source_record(
                        source=source,
                        saved=target,
                        content_type=content_type,
                        orig_name=name,
                        content_name=member,
                    )
            saved.unlink()
            if not extracted:
                raise ValueError(f"zip source did not contain XLSX/CSV data: {source['url']}")

        figure_no = int(selected["number"])
        figure_tag = f"fig_{figure_no:03d}"
        image_path = figure_dir / f"{figure_tag}.png"
        image_type = download_file(
            session,
            selected["image_url"],
            image_path,
            referer=url,
            timeout=timeout,
            sleep=sleep,
            max_retries=max_retries,
        )
        if image_type.casefold() != "image/png" or image_path.read_bytes()[:8] != b"\x89PNG\r\n\x1a\n":
            raise ValueError(f"figure endpoint did not return PNG: {selected['image_url']}")
        caption_path = figure_dir / f"{figure_tag}.txt"
        caption_path.write_text(selected["caption"] + "\n", encoding="utf-8")
        figure_records.append(
            {
                "figure_tag": figure_tag,
                "figure_no": figure_no,
                "image_file": str(out / article_id / "figures" / image_path.name),
                "caption_file": str(out / article_id / "figures" / caption_path.name),
                "image_url": selected["image_url"],
                "source_url": url,
                "panels": selected["panels"],
                "source_keys": selected.get("source_keys", []),
                "source_association": selected.get("source_association"),
            }
        )
        write_json(meta_dir / "figures.json", figure_records)
        write_json(meta_dir / "source_data.json", source_records)
        write_json(meta_dir / "_source_data_manifest.json", {"article_url": url, "links": source_records})
        require_complete_article_payload(staging)
        if base.exists():
            existing = inspect_article_payload(base)
            if existing.complete:
                shutil.rmtree(staging, ignore_errors=True)
                return True, "already-present"
            quarantine_incomplete_article(base)
        staging.replace(base)

        record = evaluate_crossref_item(item, article_html=html_text, require_cc_by=True)
        record["article_url"] = url
        record["url"] = url
        manifest = build_article_manifest(record, out, require_cc_by=True, download_status="downloaded")
        write_json(base / "meta" / "provenance.json", manifest)
        return True, "downloaded"
    except Exception:
        shutil.rmtree(staging, ignore_errors=True)
        if base.exists() and not (base / "meta" / "provenance.json").exists():
            quarantine_article_path(base, bucket="_rejected_download")
        raise


def crossref_items(
    session: requests.Session,
    *,
    issns: list[str],
    from_date: str,
    until_date: str,
    timeout: float,
    sleep: float,
    max_retries: int,
) -> Iterator[dict[str, Any]]:
    seen_dois: set[str] = set()
    for issn in issns:
        offset = 0
        first_page = True
        while offset < 10000:
            params = {
                "filter": f"issn:{issn},type:journal-article,from-pub-date:{from_date},until-pub-date:{until_date}",
                "rows": 100,
                "offset": offset,
                "mailto": CROSSREF_MAILTO,
                "select": "DOI,URL,container-title,published,published-online,published-print,issued,license,title,type",
            }
            empty_attempts = 0
            while True:
                response = request_with_retries(
                    session,
                    CROSSREF_URL,
                    params=params,
                    timeout=timeout,
                    sleep=sleep,
                    max_retries=max_retries,
                )
                items = (response.json().get("message") or {}).get("items") or []
                if items or not first_page:
                    break
                empty_attempts += 1
                if empty_attempts >= 5:
                    break
                wait = max(sleep, 1.0) * (2 ** (empty_attempts - 1))
                print(f"  [enum-retry {empty_attempts}/5] empty first page for {issn}; waiting {wait:.1f}s")
                time.sleep(wait)
            first_page = False
            if not items:
                break
            for item in items:
                doi = str(item.get("DOI") or "").casefold()
                if doi and doi in seen_dois:
                    continue
                if doi:
                    seen_dois.add(doi)
                yield item
            if len(items) < 100:
                break
            offset += 100


def validate_dates(from_date: str, until_date: str) -> None:
    try:
        start = date.fromisoformat(from_date)
        end = date.fromisoformat(until_date)
    except ValueError as exc:
        raise SystemExit(f"dates must use YYYY-MM-DD: {exc}") from exc
    if start > end:
        raise SystemExit("--from must not be later than --until")


def maybe_force_ipv4() -> bool:
    if os.environ.get("EMBO_ALLOW_IPV6") == "1":
        return False
    try:
        import urllib3.util.connection as urllib3_cn
        urllib3_cn.allowed_gai_family = lambda: socket.AF_INET
        return True
    except Exception:
        return False


def resolve_issns(args: argparse.Namespace) -> list[str]:
    selected: list[str] = []
    for issn in args.issn or []:
        if issn not in EMBO_JOURNALS:
            raise SystemExit(f"unsupported --issn {issn}; choose one of {', '.join(sorted(EMBO_JOURNALS))}")
        selected.append(issn)
    for journal in args.journal or []:
        key = journal.strip().casefold()
        if key not in JOURNAL_TO_ISSN:
            raise SystemExit(f"unsupported --journal {journal}; choose one of {', '.join(info['name'] for info in EMBO_JOURNALS.values())}")
        selected.append(JOURNAL_TO_ISSN[key])
    return _dedupe(iter(selected)) or list(EMBO_JOURNALS)


def run(args: argparse.Namespace) -> int:
    if not args.require_cc_by:
        raise SystemExit("--require-cc-by is mandatory for corpus downloads")
    if maybe_force_ipv4():
        print("[net] forcing IPv4 (set EMBO_ALLOW_IPV6=1 to allow IPv6)", flush=True)
    validate_dates(args.from_date, args.until_date)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    processed_path = out / "_processed.txt"
    skipped_path = out / "_skipped.txt"
    processed = load_processed(processed_path)
    issns = resolve_issns(args)
    admitted = 0
    examined = 0

    session = requests.Session()
    session.headers.update({"User-Agent": USER_AGENT, "Accept": "*/*", "Connection": "close"})
    print(f"[info] EMBO Crossref range {args.from_date}..{args.until_date}; issns={','.join(issns)}; output={out}")
    try:
        for item in crossref_items(
            session,
            issns=issns,
            from_date=args.from_date,
            until_date=args.until_date,
            timeout=args.timeout,
            sleep=args.sleep,
            max_retries=args.max_retries,
        ):
            doi = str(item.get("DOI") or "").casefold()
            article_id = article_id_from_doi(doi)
            if article_id is None or not is_cc_by_4(item):
                continue
            base = out / article_id
            if base.exists():
                existing = inspect_article_payload(base)
                if existing.complete:
                    print(f"[skip] {doi}: already present")
                    continue
                destination = quarantine_incomplete_article(base)
                print(
                    f"[quarantine] {doi}: {','.join(existing.reasons)} -> {destination}"
                )
                processed.discard(article_id)
            if article_id in processed:
                continue
            examined += 1
            reason = "fetch-error"
            try:
                accepted, reason = harvest_article(
                    session,
                    item,
                    out=out,
                    min_panels=args.min_panels,
                    timeout=args.timeout,
                    sleep=args.sleep,
                    max_retries=args.max_retries,
                )
            except Exception as exc:
                accepted = False
                print(f"[warn] {doi}: {type(exc).__name__}: {exc}")
            append_line(processed_path, article_id)
            processed.add(article_id)
            if accepted:
                admitted += 1
                print(f"[admit] {doi} -> {article_id}")
                if args.max_articles and admitted >= args.max_articles:
                    break
            else:
                append_line(skipped_path, f"{article_id}\t{reason}")
                print(f"[skip] {doi}: {reason}")
    finally:
        session.close()
    print(f"[done] admitted={admitted} examined={examined}")
    return 0 if admitted or not args.max_articles else 1


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=["auto"])
    parser.add_argument("--require-cc-by", action="store_true")
    parser.add_argument("--out", default="outputs/extreme_content")
    parser.add_argument("--max-articles", type=int, default=0)
    parser.add_argument("--from", dest="from_date", default=os.environ.get("PV_FROM_DATE", DEFAULT_FROM_DATE), metavar="YYYY-MM-DD")
    parser.add_argument("--until", dest="until_date", default=os.environ.get("PV_UNTIL_DATE", DEFAULT_UNTIL_DATE), metavar="YYYY-MM-DD")
    parser.add_argument("--sleep", type=float, default=0.15)
    parser.add_argument("--timeout", type=float, default=45.0)
    parser.add_argument("--max-retries", type=int, default=4)
    parser.add_argument("--min-panels", type=int, default=5)
    parser.add_argument("--issn", action="append", default=[])
    parser.add_argument("--journal", action="append", default=[])
    return parser


if __name__ == "__main__":
    raise SystemExit(run(build_parser().parse_args()))
