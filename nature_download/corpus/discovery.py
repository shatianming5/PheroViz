"""Crossref-only corpus discovery; this module never downloads article assets."""

from __future__ import annotations

from collections import Counter
from datetime import date, datetime, timezone
import json
from pathlib import Path
import time
from typing import Any, Iterable
from urllib.error import HTTPError, URLError
from urllib.parse import quote, urlencode
from urllib.request import Request, urlopen

from .policy import evaluate_crossref_item, normalize_doi


CROSSREF_WORKS_URL = "https://api.crossref.org/works"
DEFAULT_USER_AGENT = (
    "PheroViz-NatureVis2000/2.0 "
    "(CC-BY license discovery; contact via --mailto)"
)


class CrossrefError(RuntimeError):
    """Raised when Crossref cannot be queried reliably."""


def _request_json(
    url: str,
    *,
    params: dict[str, Any] | None = None,
    timeout: float = 30,
    max_retries: int = 3,
    sleep: float = 1.0,
    mailto: str | None = None,
) -> dict[str, Any]:
    query = dict(params or {})
    if mailto:
        query["mailto"] = mailto
    target = f"{url}?{urlencode(query)}" if query else url
    headers = {
        "Accept": "application/json",
        "User-Agent": DEFAULT_USER_AGENT,
    }
    last_error: Exception | None = None
    for attempt in range(max(1, max_retries)):
        try:
            request = Request(target, headers=headers)
            with urlopen(request, timeout=timeout) as response:
                payload = response.read()
            data = json.loads(payload.decode("utf-8"))
            if not isinstance(data, dict):
                raise CrossrefError("Crossref returned a non-object response")
            if sleep:
                time.sleep(sleep)
            return data
        except (HTTPError, URLError, TimeoutError, json.JSONDecodeError) as exc:
            last_error = exc
            if attempt + 1 >= max(1, max_retries):
                break
            time.sleep(max(sleep, 0.25) * (2**attempt))
    raise CrossrefError(f"Crossref request failed: {last_error}")


def crossref_discover(
    query: str,
    *,
    rows: int,
    journal: str | None = None,
    from_date: str | None = None,
    until_date: str | None = None,
    mailto: str | None = None,
    timeout: float = 30,
    max_retries: int = 3,
    sleep: float = 1.0,
    page_size: int = 250,
) -> list[dict[str, Any]]:
    if rows < 1:
        raise ValueError("rows must be positive")
    if page_size < 1 or page_size > 1000:
        raise ValueError("page_size must be between 1 and 1000")
    filters = ["type:journal-article"]
    if from_date:
        date.fromisoformat(from_date)
        filters.append(f"from-pub-date:{from_date}")
    if until_date:
        date.fromisoformat(until_date)
        filters.append(f"until-pub-date:{until_date}")

    collected: list[dict[str, Any]] = []
    cursor = "*"
    seen_cursors: set[str] = set()
    while len(collected) < rows:
        request_rows = min(page_size, rows - len(collected))
        params: dict[str, Any] = {
            "query": query,
            "filter": ",".join(filters),
            "rows": request_rows,
            "cursor": cursor,
        }
        if journal:
            params["query.container-title"] = journal
        payload = _request_json(
            CROSSREF_WORKS_URL,
            params=params,
            mailto=mailto,
            timeout=timeout,
            max_retries=max_retries,
            sleep=sleep,
        )
        message = payload.get("message", {})
        items = message.get("items", [])
        collected.extend(item for item in items if isinstance(item, dict))
        next_cursor = message.get("next-cursor") or message.get("next_cursor")
        if not items or not next_cursor or next_cursor in seen_cursors:
            break
        seen_cursors.add(cursor)
        cursor = str(next_cursor)
    return collected[:rows]


def crossref_lookup(
    doi: str,
    *,
    mailto: str | None = None,
    timeout: float = 30,
    max_retries: int = 3,
    sleep: float = 1.0,
) -> dict[str, Any]:
    normalized = normalize_doi(doi)
    if not normalized:
        raise ValueError("DOI is required")
    payload = _request_json(
        f"{CROSSREF_WORKS_URL}/{quote(normalized, safe='')}",
        mailto=mailto,
        timeout=timeout,
        max_retries=max_retries,
        sleep=sleep,
    )
    item = payload.get("message")
    if not isinstance(item, dict):
        raise CrossrefError(f"Crossref returned no work for DOI {normalized}")
    return item


def evaluate_discovery(
    items: Iterable[dict[str, Any]],
    *,
    require_cc_by: bool = True,
    retrieved_at: str | None = None,
) -> list[dict[str, Any]]:
    timestamp = retrieved_at or datetime.now(timezone.utc).replace(
        microsecond=0
    ).isoformat()
    records: list[dict[str, Any]] = []
    seen: set[str] = set()
    for item in items:
        decision = evaluate_crossref_item(
            item,
            retrieved_at=timestamp,
            require_cc_by=require_cc_by,
        )
        doi = decision.get("doi")
        if doi and doi in seen:
            continue
        if doi:
            seen.add(doi)
        records.append(decision)
    return records


def write_jsonl(path: Path, records: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(
                json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n"
            )


def write_discovery_artifacts(
    records: list[dict[str, Any]],
    output_dir: str | Path,
    *,
    query: str | None = None,
) -> dict[str, Any]:
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    accepted = [record for record in records if record.get("download_eligible")]
    rejected = [record for record in records if not record.get("download_eligible")]
    write_jsonl(output / "discovery.jsonl", records)
    write_jsonl(output / "accepted.jsonl", accepted)
    write_jsonl(output / "rejected.jsonl", rejected)
    write_jsonl(output / "download_queue.jsonl", accepted)
    reason_counts = Counter(
        reason
        for record in rejected
        for reason in record.get("reject_reasons") or ["unspecified"]
    )
    summary = {
        "query": query,
        "retrieved_at": records[0].get("retrieved_at") if records else None,
        "total": len(records),
        "accepted": len(accepted),
        "rejected": len(rejected),
        "rejection_reasons": dict(sorted(reason_counts.items())),
        "downloaded_assets": 0,
        "crossref_only": True,
    }
    (output / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return summary
