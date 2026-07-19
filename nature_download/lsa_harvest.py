#!/usr/bin/env python3
"""Harvest verified Life Science Alliance articles with per-figure source data.

Life Science Alliance (LSA) serves article HTML on the Silverchair/HighWire
platform.  A qualifying source-data file is rendered inside its owning
``div.fig`` as ``div.supplementary-material.source-data``; its download path is
``.../<article>/F<asset>/DC<n>/embed/inline-supplementary-material-<n>.xlsx``.
This harvester deliberately accepts only that figure-local shape, rather than
whole-article supplementary spreadsheets.
"""

from __future__ import annotations

import argparse
from datetime import date
import json
import os
from pathlib import Path
import re
import shutil
import socket
import time
from typing import Any, Iterator
from urllib.parse import quote, unquote, urljoin, urlparse
import zipfile

from bs4 import BeautifulSoup, Tag
from PIL import Image, UnidentifiedImageError
import requests

from corpus.policy import evaluate_crossref_item
from corpus.provenance import build_article_manifest


CROSSREF_URL = "https://api.crossref.org/works"
CROSSREF_ISSN = "2575-1077"
CROSSREF_MAILTO = "nature-vis-corpus@outlook.com"
DEFAULT_FROM_DATE = "2018-01-01"
DEFAULT_UNTIL_DATE = "2025-12-31"
LSA_HOST = "life-science-alliance.org"
USER_AGENT = (
    "PheroViz-LSA-harvester/1.0 "
    "(CC-BY research corpus; https://github.com/shatianming5/PheroViz; "
    "mailto:nature-vis-corpus@outlook.com)"
)

CC_BY_4_HTML_PATTERN = re.compile(
    r"https?://(?:www\.)?creativecommons\.org/licenses/by/4\.0(?:[/?#\"'<]|$)",
    re.I,
)
CC_BY_4_PATTERN = re.compile(
    r"^https?://(?:www\.)?creativecommons\.org/licenses/by/4\.0(?:/|$)",
    re.I,
)
DOI_PATTERN = re.compile(r"^10\.26508/lsa\.(?P<numeric_id>\d+)$", re.I)
FIGURE_CONTAINER_ID_PATTERN = re.compile(r"^F(?P<asset>\d+)$", re.I)
FIGURE_LABEL_PATTERN = re.compile(r"\b(?:figure|fig\.?)\s*(?P<number>\d+)\b", re.I)
SOURCE_PATH_PATTERN = re.compile(
    r"^/content/lsa/[^/]+/[^/]+/(?P<article>e\d+)/"
    r"F(?P<asset>\d+)/DC(?P<dc>\d+)/embed/"
    r"inline-supplementary-material-\d+\.(?P<ext>xlsx|csv)$",
    re.I,
)
IMAGE_PATH_PATTERN = re.compile(
    r"^/content/lsa/[^/]+/[^/]+/(?P<article>e\d+)/"
    r"F(?P<asset>\d+)\.large\.(?:jpe?g|png|gif|webp)$",
    re.I,
)
PANEL_GROUP_PATTERN = re.compile(
    r"\(\s*([A-L](?:\s*[,;/&+\-–—]\s*[A-L])*)\s*\)",
    re.I,
)
PANEL_SINGLE_PATTERN = re.compile(r"\(\s*([A-L])\s*\)", re.I)
TEXT_PANEL_PATTERN = re.compile(r"(?:^|[\s.;:])([A-L])\s*[,.:;]\s+", re.I)


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def load_processed(path: Path) -> set[str]:
    if not path.is_file():
        return set()
    return {
        line.strip()
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    }


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
    for attempt in range(1, max_retries + 1):
        try:
            response = session.get(
                url,
                params=params,
                headers=headers,
                timeout=(min(6.0, timeout), timeout),
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
            wait = (
                min(60.0, 8.0 * (2 ** (attempt - 1)))
                if status in (403, 429)
                else max(sleep, 0.5) * (2 ** (attempt - 1))
            )
            print(
                f"  [retry {attempt}/{max_retries}] {url}: "
                f"{type(exc).__name__}{f' {status}' if status else ''}; "
                f"waiting {wait:.1f}s",
                flush=True,
            )
            time.sleep(wait)
    raise RuntimeError("request retry loop exhausted")


def is_cc_by_4(item: dict[str, Any]) -> bool:
    return any(
        CC_BY_4_PATTERN.search(str(record.get("URL") or record.get("url") or ""))
        for record in item.get("license") or []
        if isinstance(record, dict)
    )


def has_cc_by_4_license(html_text: str) -> bool:
    """Accept only an explicit CC-BY-4.0 URL embedded in article HTML."""
    return bool(CC_BY_4_HTML_PATTERN.search(html_text))


def article_identity(doi: str) -> tuple[str, str] | None:
    """Return the on-disk DOI suffix and LSA's ``e<id>`` URL token."""
    match = DOI_PATTERN.fullmatch(doi.strip())
    if not match:
        return None
    numeric_id = match.group("numeric_id")
    return f"lsa.{numeric_id}", f"e{numeric_id}"


def article_url(doi: str) -> str:
    return (
        "https://www.life-science-alliance.org/lookup/doi/"
        f"{quote(doi, safe='/.')}"
    )


def panel_count(caption: str) -> int:
    """Conservatively infer the largest explicit alphabetical panel label."""
    letters: set[str] = set()
    for match in PANEL_GROUP_PATTERN.finditer(caption):
        letters.update(re.findall(r"[A-L]", match.group(1), flags=re.I))
    letters.update(match.group(1).upper() for match in PANEL_SINGLE_PATTERN.finditer(caption))
    text = BeautifulSoup(caption, "html.parser").get_text(" ") if "<" in caption else caption
    letters.update(match.group(1).upper() for match in TEXT_PANEL_PATTERN.finditer(text))
    return max((ord(letter) - ord("A") + 1 for letter in letters), default=0)


def _dedupe_records(records: Iterator[dict[str, str]]) -> list[dict[str, str]]:
    seen: set[str] = set()
    result: list[dict[str, str]] = []
    for record in records:
        url = record["url"]
        if url not in seen:
            seen.add(url)
            result.append(record)
    return result


def _same_lsa_host(parsed_url: Any) -> bool:
    return (
        parsed_url.scheme.casefold() == "https"
        and (parsed_url.hostname or "").casefold().removeprefix("www.") == LSA_HOST
    )


def _safe_lsa_url(href: str, article_url_value: str) -> str | None:
    absolute = urljoin(article_url_value, href)
    parsed = urlparse(absolute)
    if not _same_lsa_host(parsed):
        return None
    return parsed._replace(fragment="").geturl()


def _source_record_from_anchor(
    anchor: Tag,
    *,
    article_url_value: str,
    article_token: str,
    asset_id: str,
) -> dict[str, str] | None:
    href = str(anchor.get("href") or "")
    url = _safe_lsa_url(href, article_url_value)
    if not url:
        return None
    parsed = urlparse(url)
    match = SOURCE_PATH_PATTERN.fullmatch(parsed.path)
    if not match:
        return None
    if (
        match.group("article").casefold() != article_token.casefold()
        or match.group("asset") != asset_id
    ):
        return None
    filename = Path(unquote(parsed.path)).name
    if Path(filename).suffix.casefold() not in {".xlsx", ".csv"}:
        return None
    return {
        "url": url,
        "name": filename,
        "ext": match.group("ext").casefold(),
        "label": anchor.get_text(" ", strip=True) or filename,
        "asset_id": asset_id,
    }


def _image_url_from_container(
    container: Tag,
    *,
    article_url_value: str,
    article_token: str,
    asset_id: str,
) -> str | None:
    for anchor in container.select("a.highwire-figure-link-download[href]"):
        url = _safe_lsa_url(str(anchor.get("href") or ""), article_url_value)
        if not url:
            continue
        match = IMAGE_PATH_PATTERN.fullmatch(urlparse(url).path)
        if not match:
            continue
        if (
            match.group("article").casefold() == article_token.casefold()
            and match.group("asset") == asset_id
        ):
            return url
    return None


def parse_article_assets(
    html_text: str,
    doi: str,
    *,
    article_url_value: str,
) -> list[dict[str, Any]]:
    """Return only numbered main figures with source links in their own container.

    LSA assigns sequential DOM identifiers (``F1``, ``F2``, ...) to both main and
    supplementary figures.  The visible ``Figure N`` label and the classes
    ``type-featured``/``type-figure`` distinguish numbered main figures.  A
    source file is accepted only when its URL repeats this exact container's
    ``F<asset>`` identifier, which prevents an article-level supplement from
    being mistaken for figure-local source data.
    """
    identity = article_identity(doi)
    if identity is None:
        return []
    _, article_token = identity
    soup = BeautifulSoup(html_text, "html.parser")
    figures: list[dict[str, Any]] = []
    seen_numbers: set[int] = set()
    for container in soup.select("div.fig"):
        classes = {str(value).casefold() for value in container.get("class") or []}
        if not ({"type-featured", "type-figure"} & classes):
            continue
        container_match = FIGURE_CONTAINER_ID_PATTERN.fullmatch(
            str(container.get("id") or "")
        )
        if not container_match:
            continue
        asset_id = container_match.group("asset")
        caption_node = container.select_one(".fig-caption")
        if caption_node is None:
            continue
        caption = caption_node.get_text(" ", strip=True)
        label_node = caption_node.select_one(".fig-label")
        label = label_node.get_text(" ", strip=True) if label_node else caption
        number_match = FIGURE_LABEL_PATTERN.search(label)
        if not number_match:
            continue
        number = int(number_match.group("number"))
        if number in seen_numbers:
            continue
        source_records = _dedupe_records(
            record
            for anchor in container.select(
                "div.supplementary-material.source-data a[href]"
            )
            if (
                record := _source_record_from_anchor(
                    anchor,
                    article_url_value=article_url_value,
                    article_token=article_token,
                    asset_id=asset_id,
                )
            )
            is not None
        )
        image_url = _image_url_from_container(
            container,
            article_url_value=article_url_value,
            article_token=article_token,
            asset_id=asset_id,
        )
        seen_numbers.add(number)
        figures.append(
            {
                "number": number,
                "asset_id": asset_id,
                "caption": caption,
                "panels": panel_count(caption),
                "image_url": image_url,
                "source_records": source_records,
                "source_association": "figure-container-source-data",
            }
        )
    return sorted(figures, key=lambda figure: figure["number"])


def fetch_article_html(
    session: requests.Session,
    doi: str,
    *,
    timeout: float,
    sleep: float,
    max_retries: int,
) -> tuple[str, str]:
    response = request_with_retries(
        session,
        article_url(doi),
        accept="text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
        timeout=timeout,
        sleep=sleep,
        max_retries=max_retries,
    )
    try:
        canonical_url = response.url
        parsed = urlparse(canonical_url)
        if not _same_lsa_host(parsed):
            raise ValueError(f"article redirect left LSA host: {canonical_url}")
        return response.text, canonical_url
    finally:
        response.close()


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
        return response.headers.get("Content-Type", "").split(";", 1)[0].casefold()
    finally:
        response.close()
        if part.exists():
            part.unlink()


def _validate_source_data(path: Path, extension: str) -> None:
    if extension == "xlsx":
        if not zipfile.is_zipfile(path):
            raise ValueError(f"source endpoint did not return XLSX: {path.name}")
        return
    with path.open("rb") as handle:
        sample = handle.read(4096).lstrip().casefold()
    if not sample or sample.startswith((b"<!doctype", b"<html", b"<?xml")):
        raise ValueError(f"source endpoint did not return CSV: {path.name}")


def _convert_to_png(source_path: Path, png_path: Path) -> None:
    """Decode the LSA JPEG/GIF figure and emit a true PNG for corpus consumers."""
    try:
        with Image.open(source_path) as image:
            image.load()
            if image.mode in {"RGBA", "LA"} or "transparency" in image.info:
                converted = image.convert("RGBA")
            else:
                converted = image.convert("RGB")
            converted.save(png_path, format="PNG")
    except (OSError, UnidentifiedImageError) as exc:
        raise ValueError(f"figure endpoint did not return a decodable image: {source_path.name}") from exc
    if png_path.read_bytes()[:8] != b"\x89PNG\r\n\x1a\n":
        raise ValueError(f"figure conversion did not create PNG: {png_path.name}")


def _unique_path(directory: Path, name: str) -> Path:
    candidate = directory / name
    if not candidate.exists():
        return candidate
    stem = Path(name).stem
    suffix = Path(name).suffix
    index = 2
    while True:
        candidate = directory / f"{stem}-{index}{suffix}"
        if not candidate.exists():
            return candidate
        index += 1


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
    identity = article_identity(doi)
    if identity is None:
        return False, "invalid-doi"
    article_id, _ = identity
    html_text, canonical_url = fetch_article_html(
        session,
        doi,
        timeout=timeout,
        sleep=sleep,
        max_retries=max_retries,
    )
    if not has_cc_by_4_license(html_text):
        return False, "html-license-not-cc-by-4"
    figures = parse_article_assets(
        html_text,
        doi,
        article_url_value=canonical_url,
    )
    eligible_figures = [
        figure
        for figure in figures
        if figure["panels"] >= min_panels
        and figure.get("image_url")
        and figure.get("source_records")
    ]
    if not eligible_figures:
        return False, "no-eligible-figure-source-data"
    selected = max(
        eligible_figures,
        key=lambda figure: (int(figure["panels"]), -int(figure["number"])),
    )

    base = out / article_id
    staging = out / f".{article_id}.partial"
    shutil.rmtree(staging, ignore_errors=True)
    source_dir = staging / "source_data"
    figure_dir = staging / "figures"
    meta_dir = staging / "meta"
    source_dir.mkdir(parents=True)
    figure_dir.mkdir()
    meta_dir.mkdir()
    source_metadata: list[dict[str, Any]] = []
    try:
        for source in selected["source_records"]:
            source_name = str(source["name"])
            saved = _unique_path(source_dir, source_name)
            content_type = download_file(
                session,
                str(source["url"]),
                saved,
                referer=canonical_url,
                timeout=timeout,
                sleep=sleep,
                max_retries=max_retries,
            )
            _validate_source_data(saved, str(source["ext"]))
            source_metadata.append(
                {
                    "label": f"Life Science Alliance source data ({source['label']})",
                    "url": source["url"],
                    "saved_as": str(out / article_id / "source_data" / saved.name),
                    "saved_name": saved.name,
                    "orig_name": source_name,
                    "content_name": None,
                    "content_type": content_type,
                    "figure_no": selected["number"],
                    "figure_asset_id": selected["asset_id"],
                    "association": selected["source_association"],
                }
            )

        figure_no = int(selected["number"])
        figure_tag = f"fig_{figure_no:03d}"
        downloaded_image = figure_dir / f".{figure_tag}.source"
        image_type = download_file(
            session,
            str(selected["image_url"]),
            downloaded_image,
            referer=canonical_url,
            timeout=timeout,
            sleep=sleep,
            max_retries=max_retries,
        )
        if not image_type.startswith("image/"):
            raise ValueError(
                f"figure endpoint did not return an image: {selected['image_url']}"
            )
        image_path = figure_dir / f"{figure_tag}.png"
        _convert_to_png(downloaded_image, image_path)
        downloaded_image.unlink()
        caption_path = figure_dir / f"{figure_tag}.txt"
        caption_path.write_text(str(selected["caption"]) + "\n", encoding="utf-8")
        figure_metadata = [
            {
                "figure_tag": figure_tag,
                "figure_no": figure_no,
                "image_file": str(out / article_id / "figures" / image_path.name),
                "caption_file": str(out / article_id / "figures" / caption_path.name),
                "image_url": selected["image_url"],
                "source_url": canonical_url,
                "panels": selected["panels"],
                "figure_asset_id": selected["asset_id"],
                "source_association": selected["source_association"],
                "source_urls": [
                    source["url"] for source in selected["source_records"]
                ],
            }
        ]
        write_json(meta_dir / "figures.json", figure_metadata)
        write_json(meta_dir / "source_data.json", source_metadata)
        write_json(
            meta_dir / "_source_data_manifest.json",
            {"article_url": canonical_url, "links": source_metadata},
        )
        if base.exists():
            shutil.rmtree(staging, ignore_errors=True)
            return True, "already-present"
        staging.replace(base)

        record = evaluate_crossref_item(
            item,
            article_html=html_text,
            require_cc_by=True,
        )
        record["article_url"] = canonical_url
        record["url"] = canonical_url
        if not record["download_eligible"]:
            raise ValueError(
                "policy rejected verified LSA article: "
                + ",".join(record.get("reject_reasons") or ["unknown"])
            )
        manifest = build_article_manifest(
            record,
            out,
            require_cc_by=True,
            download_status="downloaded",
        )
        write_json(base / "meta" / "provenance.json", manifest)
        return True, "downloaded"
    except Exception:
        shutil.rmtree(staging, ignore_errors=True)
        if base.exists() and not (base / "meta" / "provenance.json").exists():
            shutil.rmtree(base, ignore_errors=True)
        raise


def crossref_items(
    session: requests.Session,
    *,
    from_date: str,
    until_date: str,
    timeout: float,
    sleep: float,
    max_retries: int,
) -> Iterator[dict[str, Any]]:
    """Enumerate the ISSN with offset pagination (LSA remains below 10,000 works)."""
    seen_dois: set[str] = set()
    offset = 0
    first_page = True
    while offset < 10_000:
        params = {
            "filter": (
                f"issn:{CROSSREF_ISSN},type:journal-article,"
                f"from-pub-date:{from_date},until-pub-date:{until_date}"
            ),
            "rows": 100,
            "offset": offset,
            "sort": "published",
            "order": "asc",
            "mailto": CROSSREF_MAILTO,
            "select": (
                "DOI,URL,container-title,published,published-online,"
                "published-print,issued,license,title,type"
            ),
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
            try:
                items = (response.json().get("message") or {}).get("items") or []
            finally:
                response.close()
            if items or not first_page:
                break
            empty_attempts += 1
            if empty_attempts >= 5:
                break
            wait = max(sleep, 1.0) * (2 ** (empty_attempts - 1))
            print(
                f"  [enum-retry {empty_attempts}/5] empty first page for "
                f"{from_date}..{until_date}; waiting {wait:.1f}s",
                flush=True,
            )
            time.sleep(wait)
        first_page = False
        if not items:
            return
        for item in items:
            doi = str(item.get("DOI") or "").casefold()
            if doi and doi in seen_dois:
                continue
            if doi:
                seen_dois.add(doi)
            yield item
        if len(items) < 100:
            return
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
    """Use IPv4 unless explicitly opted out for flaky IPv6 CDN routes."""
    if os.environ.get("LSA_ALLOW_IPV6") == "1":
        return False
    try:
        import urllib3.util.connection as urllib3_cn

        urllib3_cn.allowed_gai_family = lambda: socket.AF_INET
        return True
    except Exception:
        return False


def run(args: argparse.Namespace) -> int:
    if not args.require_cc_by:
        raise SystemExit("--require-cc-by is mandatory for corpus downloads")
    if maybe_force_ipv4():
        print("[net] forcing IPv4 (set LSA_ALLOW_IPV6=1 to allow IPv6)", flush=True)
    validate_dates(args.from_date, args.until_date)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    processed_path = out / "_lsa_processed.txt"
    skipped_path = out / "_lsa_skipped.txt"
    processed = load_processed(processed_path)
    admitted = 0
    examined = 0

    session = requests.Session()
    session.headers.update(
        {"User-Agent": USER_AGENT, "Accept": "*/*", "Connection": "close"}
    )
    print(
        f"[info] LSA Crossref range {args.from_date}..{args.until_date}; "
        f"issn={CROSSREF_ISSN}; output={out}",
        flush=True,
    )
    try:
        for item in crossref_items(
            session,
            from_date=args.from_date,
            until_date=args.until_date,
            timeout=args.timeout,
            sleep=args.sleep,
            max_retries=args.max_retries,
        ):
            doi = str(item.get("DOI") or "").casefold()
            identity = article_identity(doi)
            if identity is None or not is_cc_by_4(item):
                continue
            article_id, _ = identity
            if (out / article_id).is_dir():
                print(f"[skip] {doi}: already present", flush=True)
                continue
            if article_id in processed:
                continue
            examined += 1
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
                # Do not mark transient failures as processed: a resumed run can
                # retry them without treating unverified content as a hard skip.
                reason = f"fetch-error:{type(exc).__name__}"
                append_line(skipped_path, f"{article_id}\t{reason}")
                print(f"[warn] {doi}: {type(exc).__name__}: {exc}", flush=True)
                continue
            append_line(processed_path, article_id)
            processed.add(article_id)
            if accepted:
                admitted += 1
                print(f"[admit] {doi} -> {article_id}", flush=True)
                if args.max_articles and admitted >= args.max_articles:
                    break
            else:
                append_line(skipped_path, f"{article_id}\t{reason}")
                print(f"[skip] {doi}: {reason}", flush=True)
    finally:
        session.close()
    print(f"[done] admitted={admitted} examined={examined}", flush=True)
    return 0 if admitted or not args.max_articles else 1


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=["auto"])
    parser.add_argument("--require-cc-by", action="store_true")
    parser.add_argument("--out", default="outputs/extreme_content")
    parser.add_argument("--max-articles", type=int, default=0)
    parser.add_argument(
        "--from",
        dest="from_date",
        default=os.environ.get("PV_FROM_DATE", DEFAULT_FROM_DATE),
        metavar="YYYY-MM-DD",
    )
    parser.add_argument(
        "--until",
        dest="until_date",
        default=os.environ.get("PV_UNTIL_DATE", DEFAULT_UNTIL_DATE),
        metavar="YYYY-MM-DD",
    )
    parser.add_argument("--sleep", type=float, default=0.5)
    parser.add_argument("--timeout", type=float, default=45.0)
    parser.add_argument("--max-retries", type=int, default=4)
    parser.add_argument("--min-panels", type=int, default=5)
    return parser


if __name__ == "__main__":
    raise SystemExit(run(build_parser().parse_args()))
