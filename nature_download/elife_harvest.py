#!/usr/bin/env python3
"""Harvest CC-BY eLife articles with XLSX source data and >=5-panel figures."""

from __future__ import annotations

import argparse
from datetime import date, datetime, timezone
import json
import os
from pathlib import Path
import re
import shutil
import time
from typing import Any, Iterator
from urllib.parse import unquote, urljoin, urlparse
import zipfile

from bs4 import BeautifulSoup
import requests

from corpus.policy import evaluate_crossref_item
from corpus.provenance import build_article_manifest


CROSSREF_URL = "https://api.crossref.org/journals/2050-084X/works"
DEFAULT_FROM_DATE = "2013-01-01"
DEFAULT_UNTIL_DATE = "2022-12-31"
USER_AGENT = (
    "PheroViz-eLife-harvester/1.0 "
    "(CC-BY research corpus; https://github.com/shatianming5/PheroViz)"
)
CC_BY_4_PATTERN = re.compile(
    r"^https?://(?:www\.)?creativecommons\.org/licenses/by/4\.0(?:/|$)",
    re.I,
)
DOI_PATTERN = re.compile(r"^10\.7554/elife\.(\d+)$", re.I)
SOURCE_DATA_PATTERN = re.compile(
    r"^/articles/(?P<id>\d+)/elife-(?P=id)-"
    r"(?:[^/?#]+-)?data\d+(?:-[^/?#]+)*\.xlsx$",
    re.I,
)
MAIN_FIGURE_PATTERN = re.compile(
    r"elife-(?P<id>\d+)-fig(?P<number>\d+)-v\d+\.tiff?(?:/|$)",
    re.I,
)
PANEL_GROUP_PATTERN = re.compile(
    r"\(\s*([A-L](?:\s*[,;/&+\-–—]\s*[A-L])*)\s*\)",
    re.I,
)


def utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


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
    timeout: float,
    sleep: float,
    max_retries: int,
    stream: bool = False,
) -> requests.Response:
    headers = {"Referer": referer} if referer else None
    for attempt in range(1, max_retries + 1):
        try:
            response = session.get(
                url,
                params=params,
                headers=headers,
                timeout=timeout,
                stream=stream,
            )
            response.raise_for_status()
            if sleep:
                time.sleep(sleep)
            return response
        except requests.RequestException as exc:
            if attempt == max_retries:
                raise
            wait = max(sleep, 0.5) * (2 ** (attempt - 1))
            print(
                f"  [retry {attempt}/{max_retries}] {url}: "
                f"{type(exc).__name__}; waiting {wait:.1f}s"
            )
            time.sleep(wait)
    raise RuntimeError("request retry loop exhausted")


def crossref_items(
    session: requests.Session,
    *,
    from_date: str,
    until_date: str,
    timeout: float,
    sleep: float,
    max_retries: int,
) -> Iterator[dict[str, Any]]:
    cursor = "*"
    while cursor:
        params = {
            "filter": (
                f"from-pub-date:{from_date},until-pub-date:{until_date}"
            ),
            "rows": 100,
            "cursor": cursor,
            "sort": "published",
            "order": "desc",
            "select": (
                "DOI,URL,container-title,published,published-online,"
                "published-print,issued,license,title,type"
            ),
        }
        response = request_with_retries(
            session,
            CROSSREF_URL,
            params=params,
            timeout=timeout,
            sleep=sleep,
            max_retries=max_retries,
        )
        message = response.json().get("message") or {}
        items = message.get("items") or []
        if not items:
            return
        yield from items
        next_cursor = message.get("next-cursor")
        if not next_cursor or next_cursor == cursor:
            return
        cursor = str(next_cursor)


def is_cc_by_4(item: dict[str, Any]) -> bool:
    return any(
        CC_BY_4_PATTERN.search(str(record.get("URL") or ""))
        for record in item.get("license") or []
        if isinstance(record, dict)
    )


def article_identity(doi: str) -> tuple[str, str] | None:
    match = DOI_PATTERN.fullmatch(doi.strip())
    if not match:
        return None
    numeric_id = str(int(match.group(1)))
    return numeric_id, f"elife-{numeric_id}"


def source_data_urls(
    soup: BeautifulSoup, page_url: str, numeric_id: str
) -> list[str]:
    urls: set[str] = set()
    for anchor in soup.select("a[href]"):
        url = urljoin(page_url, str(anchor.get("href") or ""))
        parsed = urlparse(url)
        if (
            parsed.hostname
            and parsed.hostname.casefold() == "cdn.elifesciences.org"
            and (match := SOURCE_DATA_PATTERN.fullmatch(unquote(parsed.path)))
            and match.group("id") == numeric_id
        ):
            urls.add(parsed._replace(query="", fragment="").geturl())
    return sorted(urls)


def panel_count(caption: str) -> int:
    letters: set[str] = set()
    for match in PANEL_GROUP_PATTERN.finditer(caption):
        letters.update(re.findall(r"[A-L]", match.group(1), flags=re.I))
    return max((ord(letter.upper()) - ord("A") + 1 for letter in letters), default=0)


def full_size_png_url(url: str) -> str:
    if urlparse(url).hostname == "iiif.elifesciences.org":
        return re.sub(
            r"/full/[^/]+/0/default\.(?:jpe?g|png)(?:\?.*)?$",
            "/full/full/0/default.png",
            url,
            flags=re.I,
        )
    return url


def main_figures(
    soup: BeautifulSoup, page_url: str, numeric_id: str
) -> list[dict[str, Any]]:
    figures: list[dict[str, Any]] = []
    seen: set[int] = set()
    for element in soup.select("figure"):
        image = element.select_one("img[src]")
        caption_element = element.select_one("figcaption")
        if image is None or caption_element is None:
            continue
        image_url = urljoin(page_url, str(image.get("src") or ""))
        image_path = unquote(urlparse(image_url).path)
        match = MAIN_FIGURE_PATTERN.search(image_path)
        if (
            not match
            or match.group("id") != numeric_id
            or int(match.group("number")) in seen
        ):
            continue
        number = int(match.group("number"))
        seen.add(number)
        caption = caption_element.get_text(" ", strip=True)
        figures.append(
            {
                "number": number,
                "caption": caption,
                "panels": panel_count(caption),
                "image_url": full_size_png_url(image_url),
            }
        )
    return sorted(figures, key=lambda figure: figure["number"])


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


def harvest_article(
    session: requests.Session,
    item: dict[str, Any],
    *,
    out: Path,
    timeout: float,
    sleep: float,
    max_retries: int,
) -> tuple[bool, str]:
    doi = str(item.get("DOI") or "").casefold()
    identity = article_identity(doi)
    if identity is None:
        return False, "invalid-doi"
    numeric_id, article_id = identity
    article_url = f"https://elifesciences.org/articles/{numeric_id}"
    figures_url = f"{article_url}/figures"
    response = request_with_retries(
        session,
        figures_url,
        timeout=timeout,
        sleep=sleep,
        max_retries=max_retries,
    )
    soup = BeautifulSoup(response.text, "html.parser")
    sources = source_data_urls(soup, response.url, numeric_id)
    if not sources:
        return False, "no-source-data"
    figures = main_figures(soup, response.url, numeric_id)
    if not figures:
        return False, "no-figures"
    eligible_figures = [figure for figure in figures if figure["panels"] >= 5]
    if not eligible_figures:
        return False, "insufficient-panels"
    selected = max(
        eligible_figures,
        key=lambda figure: (figure["panels"], -figure["number"]),
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
    source_records: list[dict[str, Any]] = []
    figure_records: list[dict[str, Any]] = []
    try:
        for source_url in sources:
            name = Path(unquote(urlparse(source_url).path)).name
            saved = source_dir / name
            content_type = download_file(
                session,
                source_url,
                saved,
                referer=response.url,
                timeout=timeout,
                sleep=sleep,
                max_retries=max_retries,
            )
            if not zipfile.is_zipfile(saved):
                raise ValueError(f"download is not a valid XLSX archive: {source_url}")
            source_records.append(
                {
                    "label": f"eLife source data ({name})",
                    "url": source_url,
                    "saved_as": str(out / article_id / "source_data" / name),
                    "saved_name": name,
                    "orig_name": name,
                    "content_name": None,
                    "content_type": content_type,
                }
            )

        figure_no = int(selected["number"])
        figure_tag = f"fig_{figure_no:03d}"
        image_path = figure_dir / f"{figure_tag}.png"
        image_type = download_file(
            session,
            selected["image_url"],
            image_path,
            referer=response.url,
            timeout=timeout,
            sleep=sleep,
            max_retries=max_retries,
        )
        if image_type.casefold() != "image/png" or image_path.read_bytes()[:8] != (
            b"\x89PNG\r\n\x1a\n"
        ):
            raise ValueError(
                f"figure endpoint did not return PNG: {selected['image_url']}"
            )
        caption_path = figure_dir / f"{figure_tag}.txt"
        caption_path.write_text(selected["caption"] + "\n", encoding="utf-8")
        figure_records.append(
            {
                "figure_tag": figure_tag,
                "figure_no": figure_no,
                "image_file": str(
                    out / article_id / "figures" / image_path.name
                ),
                "caption_file": str(
                    out / article_id / "figures" / caption_path.name
                ),
                "image_url": selected["image_url"],
                "source_url": figures_url,
            }
        )
        write_json(meta_dir / "figures.json", figure_records)
        write_json(meta_dir / "source_data.json", source_records)
        write_json(
            meta_dir / "_source_data_manifest.json",
            {"article_url": article_url, "links": source_records},
        )
        if base.exists():
            shutil.rmtree(staging, ignore_errors=True)
            return True, "already-present"
        staging.replace(base)

        record = evaluate_crossref_item(item, require_cc_by=True)
        record["article_url"] = article_url
        record["url"] = article_url
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


def validate_dates(from_date: str, until_date: str) -> None:
    try:
        start = date.fromisoformat(from_date)
        end = date.fromisoformat(until_date)
    except ValueError as exc:
        raise SystemExit(f"dates must use YYYY-MM-DD: {exc}") from exc
    if start > end:
        raise SystemExit("--from must not be later than --until")


def run(args: argparse.Namespace) -> int:
    if not args.require_cc_by:
        raise SystemExit("--require-cc-by is mandatory for corpus downloads")
    validate_dates(args.from_date, args.until_date)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    processed_path = out / "_processed.txt"
    skipped_path = out / "_skipped.txt"
    processed = load_processed(processed_path)
    admitted = 0
    examined = 0

    session = requests.Session()
    session.headers.update(
        {
            "User-Agent": USER_AGENT,
            "Accept": "*/*",
            "Connection": "close",
        }
    )
    print(
        f"[info] eLife Crossref range {args.from_date}..{args.until_date}; "
        f"output={out}"
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
            _, article_id = identity
            if (out / article_id).is_dir():
                print(f"[skip] {doi}: already present")
                continue
            if article_id in processed:
                continue
            examined += 1
            reason = "fetch-error"
            try:
                accepted, reason = harvest_article(
                    session,
                    item,
                    out=out,
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
    parser.add_argument("--sleep", type=float, default=0.15)
    parser.add_argument("--timeout", type=float, default=45.0)
    parser.add_argument("--max-retries", type=int, default=4)
    return parser


if __name__ == "__main__":
    raise SystemExit(run(build_parser().parse_args()))
