Nature Family Search (Compliant, API‑based)

Overview
- This project provides a minimal, compliant workflow to search Nature family journal articles, collect metadata and abstracts via official/allowed APIs, and optionally download figures only for open-access (PMC) articles.
- It avoids bypassing anti-bot/anti-scraping protections and does not automate downloads from publisher sites that disallow it. It relies on Crossref + Europe PMC (and PMC) which are intended for programmatic use.

What it does
- Search Crossref for Springer Nature journal articles; filter to titles beginning with "Nature" (e.g., Nature, Nature Medicine, Nature Physics, etc.).
- Resolve metadata and abstracts via Crossref and Europe PMC (when available).
- For articles that have a PMC ID (open access), fetch figure image URLs from the PMC article page and download locally with polite throttling.
- Save results into a structured JSONL/CSV under `outputs/`.
 - Include `authors` (Crossref preferred, Europe PMC fallback) and `pmid`; supports Chinese keywords and safe Windows console output.

What it does NOT do
- No anti-bot evasion, no captcha workarounds, and no scraping of publisher pages that disallow it.
- No download of closed-access content. Images are only fetched for OA PMC articles.

Quick start
- Python 3.9+ recommended.
- Run: `python scripts/nature_cli.py --query "cancer" --max 5 --images --out outputs/sample`
- The CLI installs minimal dependencies (requests, beautifulsoup4) if missing.
- Options:
  - `--timeout 30` and `--max-retries 3` for robust retries
  - `--append` to append JSONL with DOI de-duplication
  - `--no-family-bias` to remove `query.container-title=Nature` bias
  - `--mailto you@example.com` to include a Crossref mailto identifier

Notes on compliance
- Crossref and Europe PMC are designed for API usage. Provide a clear User-Agent and throttle requests.
- Check licenses for reuse of images and data; open access does not always mean images may be reused.
- Respect rate limits. Defaults are very conservative (1 request/sec).

Git workflow
- Each change is committed; pushing requires that `origin` is configured with valid credentials.
- A simple `VERSION_LOG.md` records versions and change notes.

Figure fetch (authorized)
- If you have authorization to fetch figure pages directly from nature.com, use:
- `python scripts/nature_fig_fetch.py --url "https://www.nature.com/articles/<article-id>/figures/1" --out outputs/nature_figs`
- Saves figure image and `*_caption.txt` plus a small `*_meta.json` with source URL and caption.

Source data fetch (authorized)
- If you have authorization to fetch "Source data" attachments directly from nature.com, use:
- `python scripts/nature_source_data_fetch.py --url "https://www.nature.com/articles/<article-id>#Sec71" --out outputs/nature_source_data`
- Optional filters:
  - `--section-id Sec71` only search within a section id
  - `--filter "Fig. 4"` narrow by label (e.g., Fig. 4, Extended Data Fig. 7)
- Saves downloaded files and a `_source_data_manifest.json` with labels and URLs.
