# C2 候选池扩量（A3，纯离线）

## 已交叉验证的结论

- `nature_content` 的“90”是顶层条目数，不是 90 个完整 DOI：实际有 87 个非私有目录、79 个带 provenance 的 article dir，只有 **59 DOI** 同时满足 PNG + XLSX + 三份 meta。
- A3 物化前，所有既有 `outputs` 中有 **3829 个 distinct DOI** 满足同一完整判据（6674 个 dir）；本地有 **24,537** 个 XLSX（含解压/重复位置），其中路径含 `source_data` 目录组件的有 **23,027** 个。

| `outputs` 子目录 | provenance article dir | 完整 DOI（PNG+XLSX+meta） | XLSX |
|---|---:|---:|---:|
| `extreme_content` | 3,656 | 3,421 | 10,144 |
| `extreme_merged` | 3,408 | 3,175 | 12,309 |
| `nature_content` | 79 | 59 | 60 |
| `elife_2020_2026` | 19 | 19 | 160 |

各根之间存在 DOI 重叠，不能相加为全局 DOI 数。

- 对当前全物化 raw 输入（5,377 candidates）重新运行 `propose-cases`：**32** strict multi、**32 DOI**；≥P5 为 **7** 个 / **7 DOI**（P5=2, P6=1, P7=1, P8=1, P11=1, P13=1）。这复现了现有的约 32 条 strict multi，不应误报为 32 DOI。
- 对冻结 `c2_cases_v1` 的 **2,544** 个 byte-pinned 工作表重新运行同一 strict proposer：**487 strict multi**，覆盖 **427 distinct DOI**；其中 ≥P5 **224** 个 figure case、**204 DOI**。这是当前可送外部审查的**理论 DOI 天花板**（全部 `eligible_for_experiment=false`，不是 verified）。
- ≥P5 面板分布：**P5=65, P6=51, P7=30, P8=33, P9=11, P10=10, P11=6, P12=8, P13=4, P14=1, P15=2, P16=2, P17=1**。P12 及以上也保留，未被截断。

## 物化与成本

`c2_cases_v1` 有 588 DOI / 680 figure group（P5+：204 DOI / 224 group）。其 2,544 个原始表和工作表均已逐字节复核；全部 680 个对应 Figure PNG 也已按 provenance SHA-256 复核。

- 原先在单一 article dir 中完整（PNG+XLSX+meta）：**558/588 DOI**，P5+ **192/204 DOI**。
- 对缺失的 30 DOI，已在 `nature_download/outputs/c2_widen_pool_overlay` 用本地硬链接重组；图和 meta 来自现有 article dir，原始 Source Data 来自本地已解压表。**未下载、未调用 gateway、未用 symlink**。重组后 XLSX 严格判据覆盖 **583/588 DOI**，P5+ **203/204 DOI**。
- 剩余差额是 **7 个 CSV-only DOI**（冻结原始表格式为 CSV，未伪造转换成 XLSX）；按“本地原始表（XLSX/CSV）+ 工作表 + PNG + meta”判据则为 **588/588 DOI**、P5+ **204/204 DOI**。因此扩量无需重新下载，成本只是本地重组/路径索引。

## 严格性与图型边界

宽池输出的 constituent panel chart family 为：bar=2267, line=84。所有 **487** 个 strict multi 的所有面板均为 `line`/`bar`/`scatter`；含其他 family 的已提出 multi 为 **0**。但 proposer 对被拒输入只给出表格契约/可渲染性拒因，并不进行图像语义“其他图型”标注；因此不能把 0 误解为整个语料中其他图型为 0。

重要限定：该宽池工作表包含 **141 A_raw + 2,403 B_normalizer**。`strict` 指当前 fail-closed proposer 已对 byte-pinned 工作表重新判定并通过 renderability audit；它**不**把 B_normalizer 变成 raw-only，也**不**把任何案例变成 verified。multi 的 tier 构成：A_raw=9, A_raw+B_normalizer=45, B_normalizer=433。

## 产物

- `files/c2_widen_pool/materialization_census.json` — 完整普查、路径、哈希与两次 strict run 的 JSONL 交叉核对。
- `files/c2_widen_pool/strict_multi_candidates.jsonl` — 487 行；是 frozen working-table proposer 输出的逐行过滤子集。
- `files/c2_widen_pool/pool_ceiling.md` — 本报告。
- 可复现输入/完整输出：`files/c2_widen_pool/c2_cases_v1_working_candidates.jsonl`、`files/c2_widen_pool/frozen_working_table_strict_proposer/`、`files/c2_widen_pool/raw_materialized_strict_proposer/`。本地重组目录位于 gitignored 的 `nature_download/outputs/c2_widen_pool_overlay`。

所有 DOI 数、multi 数和面板分布均由本次产出的 JSONL 用 Python 重算并与 `summary.json` 断言一致。
