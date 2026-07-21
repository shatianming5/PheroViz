# Diagnostic V4 P5+ multi-panel ceiling — sealed-path blocked

- **Hard blocker:** the former 204-DOI V4 pool must **not** be used for sealed C2 review, benchmark construction, or K=62 readiness. It is structurally V4-parity-valid but violates the no-exploratory-normalizer red line.
- **Python audit:** **1538/1611** component singles originate from `exploratory_normalizer`, confirmed identically by `data/c2_cases_v1/_manifest.jsonl` as `normalize_source_sheet`; **223/224** multis and **203/204** DOI contain at least one affected component.
- **Former 204-DOI ceiling:** diagnostic only. The only all-direct projection inside that artifact is one P11 multi / one DOI (`10.1038/s44319-025-00503-8`).
- **Disqualified frozen copy:** `frozen_v4_full_nonoverlap_20260721T191806Z/v4_full_nonoverlap.proposed.jsonl` remains immutable (`6486ed743344731cc00202427bd722cd16dc05a69507fca887c235a1c9766d6d`) but must not be sent to sealed review.
- **Task-linked strict raw diagnostic freeze:** `files/c2_reclassify_raw_p5/frozen_raw_p5_v4_20260721T192239Z/raw_p5_v4.proposed.jsonl` (`4d0edd9bbb4e3a1883535b40867cdd383767155d8fb612ad1caaede2ec8bc6e2`) has 55 direct, hash-bound singles and 7 non-overlapping P5 parents / 7 DOI; its materialization-census source hash is verified; zero normalizer metadata/path; offline V4 preflight passed.
- **Additional direct-raw diagnostic freeze:** `files/c2_reclassify_raw_p5/frozen_direct_raw_p5_v4_20260721T193342Z/direct_raw_p5_v4.proposed.jsonl` (`d7c655b97334186d75c99ff69e23615065535224e609b40e62e1c50e26cde801`) has 124 direct, hash-bound singles and 15 P5 parents / 15 DOI; also zero normalizer metadata/path and offline-preflight-valid.
- **K=62 failure:** neither diagnostic pool is a sealed replacement: 7 DOI is 55 short and 15 DOI is 47 short of K=62; both have 0 fresh review-bound verified DOI.
- **Priority-review clarification:** a new immutable V3 priority-input binding for `c3ea…` / `737c…` exists, but it is explicitly historic-reject-selected diagnostic scope and requires fresh review; it is neither review evidence nor a V4/full-pool binding, so it cannot alter this sealed blocker.
- **Offline limit:** the frozen `data/c2_p5_4k` definition names 4,000 articles but has `content_materialized=false` and zero local article/source/figure assets, so this offline task cannot materialize the larger raw replacement.
- **Required remediation:** materialize and hash-bind a larger raw, non-normalized corpus; generate candidates without normalization; re-propose source-structure-only; freeze a >=62-potential-DOI pool; then obtain fresh dual-judge review/evidence.

Details and reproducible counts: `normalizer_redline_audit.json`, `raw_non_normalized_feasibility_probe.json`, and `SEALED_C2_BLOCKED_normalizer_pool.json`.
