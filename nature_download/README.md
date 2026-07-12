# Nature-Vis2000 CC-BY corpus gate

## 许可边界
- 允许期刊仅为 **Nature Communications**、**Scientific Reports** 和标题以 `npj ` 开头的 npj 系列。
- Nature 主刊、未知期刊和其他 Nature-family 期刊一律 fail closed。
- 仅接受文章 **Version of Record (VOR)** 或文章页元数据中可核验的 CC BY 3.0/4.0 URL。仅 TDM 为 CC BY、VOR 为 CC BY-NC/SA/ND、仅有 OA 标志或缺少 license 均拒绝。
- `is_open_access` 仅作信息字段，绝不作为再分发许可。
- 所有下载入口都要求 `--require-cc-by`；新 corpus 子命令默认开启该 gate。

## 核心能力
- **许可发现**：Crossref 检索后逐条输出接受/拒绝判定、规范化 license、证据和拒绝原因，不下载文章资产。
- **检索聚合**：旧 `search` 命令保留兼容性，并补充可执行的许可判定。
- **图像抓取**：解析 `https://www.nature.com/articles/<article-id>/figures/<n>`，提取原图 URL、caption，自动清理无图页面、连续空页可控。
- **Source data 下载**：解析文章页的 “Source data” 区块，逐个保存附件并生成 manifest，支持大文件流式下载与超时控制。
- **Provenance**：记录 DOI、期刊、年份、文章 URL、license 来源/证据/生效日、抓取时间、每个文件的 SHA-256、来源 URL、状态和拒绝原因。
- **Paper-level split**：固定 seed、按 DOI 去重、保证 train/val/test 不交叉，记录源 manifest SHA-256；model cutoff 只能由 `--model-cutoff-date` 显式传入。
- **Benchmark candidates**：安全展开 Source Data ZIP，仅识别 CSV/XLSX，并按文件名保守关联 figure/panel；自动候选永远是 `unverified`。

## 快速开始
- 使用 Python 3.10+；运行时不会安装依赖，缺失依赖会明确退出：
  ```bash
  python -m pip install -r requirements.txt
  # tests:
  python -m pip install -r requirements-dev.txt
  ```
- 仅发现并验证许可（不下载文章内容）：
  ```bash
  python nature_all_in_one.py discover \
    --journal "Nature Communications" \
    --from-date 2024-01-01 --until-date 2026-07-12 --max 2000 \
    --out outputs/corpus_discovery
  ```
- 重验已有 JSONL：
  ```bash
  python nature_all_in_one.py validate \
    --jsonl outputs/corpus_discovery/discovery.jsonl \
    --out outputs/corpus_validation
  ```
- 对已授权下载目录构建 provenance manifest：
  ```bash
  python nature_all_in_one.py build-manifest \
    --jsonl outputs/corpus_validation/discovery.jsonl \
    --content-root outputs/nature_content \
    --out outputs/corpus_manifest
  ```
- 生成 DOI-disjoint splits；cutoff 不提供时 strata 明确记为 `unconfigured`：
  ```bash
  python nature_all_in_one.py split \
    --manifest outputs/corpus_manifest/corpus_manifest.jsonl \
    --seed 20260712 \
    --model-cutoff-date "<verified-model-cutoff-date>" \
    --out outputs/corpus_splits
  ```
  `--model-cutoff-date` 必须来自对应模型供应方的可核验文档；未知时省略，输出会明确记录为 `unconfigured`。
- 从已下载 Source Data 构建 fail-closed case skeleton（不调用 LLM）：
  ```bash
  python nature_all_in_one.py build-cases \
    --corpus-manifest outputs/corpus_manifest/corpus_manifest.jsonl \
    --content-root outputs/nature_content \
    --out outputs/benchmark_candidates
  ```
  输出 `candidates.jsonl`、`ambiguous.jsonl`、`summary.json`。未提供有效
  `--evidence` 时所有候选均为 `unverified` 且
  `eligible_for_experiment=false`。ZIP 默认最多 1,000 个文件、512 MiB
  总展开大小，并拒绝绝对路径、zip-slip、symlink、特殊文件和加密成员。
  对 generic XLSX 会以 openpyxl `read_only` 仅读取 sheet 名，并从明确的
  `Fig. 1a` / `Figure 3C` 名称映射 panel；默认最多 256 sheets。无 panel、
  panel range/list、resource fork、supplementary 或损坏 workbook 均进入
  `ambiguous.jsonl`。
- 从 `unverified` candidates 生成 deterministic case proposal：
  ```bash
  python nature_all_in_one.py propose-cases \
    --candidates outputs/benchmark_candidates/candidates.jsonl \
    --out outputs/case_proposals
  ```
  仅接受 2--6 列的简单二维 CSV 或指定 XLSX sheet；默认限制 64 MiB、
  100,000 行和 64 列。输出 `proposed.jsonl`、`rejected.jsonl`、
  `summary.json`。proposal 始终为 `curation_status=proposed`、
  `eligible_for_experiment=false`，必须经外部或人工验证后才能进入实验。
- 提交并确认工作树干净后，使用至少两个不同模型做外部验证：
  ```bash
  python nature_all_in_one.py review-proposals \
    --proposed outputs/case_proposals/proposed.jsonl \
    --judge-model "<judge-model-a>" \
    --judge-model "<judge-model-b>" \
    --out outputs/case_reviews
  ```
  模型连接从既有 `MODEL_API_BASE` / `MODEL_API_KEY`（或兼容环境变量）
  读取，secret 不写入产物。输出 `evidence.json`、`reviews.jsonl`、
  `rejected.jsonl`、`summary.json` 与 `summary.sha256`。默认拒绝 dirty
  worktree；`--resume` 仅复用与输入、资产、rubric、模型和代码 hash
  完全绑定的 sidecar。
- 基础检索（合规、不抓取）：
  ```bash
  python nature_all_in_one.py search \
    --query "cancer" --max 5 --require-cc-by \
    --out outputs/search_run
  ```
  常用可选项：`--timeout 30`、`--max-retries 3`、`--append`、`--no-family-bias`、`--mailto you@example.com`。

## 全功能入口 `nature_all_in_one.py`
- 仓库已将全部能力整合在单脚本中，可通过子命令组合使用。
- 自动搜索 + 抓取全部（图像 + caption + Source data）：
  - **两阶段（先搜后抓）**
    ```bash
    python nature_all_in_one.py auto --require-cc-by --max-per-keyword 50 --max-articles 200 --max-figs 12 --sort year_desc
    ```
  - **流式模式（边搜边抓）**
    ```bash
    python nature_all_in_one.py auto --require-cc-by --stream --stream-workers 6 --max-per-keyword 50 --max-articles 200 --max-figs 12
    ```
    `--stream-workers` 控制抓取线程数（默认 1），Rich 会显示关键词进度、成功数量与各 worker 状态。
  - 其他常用可选项：`--keywords-file keywords.txt`（每行一个关键词）、`--mailto you@example.com`、`--sleep 1.0`、`--timeout 300`、`--max-retries 5`。
- 仅检索：`python nature_all_in_one.py search --query "cancer" --max 20 --out outputs/search_run --append --require-cc-by`
- 仅抓图：`python nature_all_in_one.py fig --require-cc-by --doi "<doi>" --url "https://www.nature.com/articles/<article-id>/figures/1" --out outputs/nature_content`
- 仅抓 Source data：`python nature_all_in_one.py source --require-cc-by --doi "<doi>" --url "https://www.nature.com/articles/<article-id>" --out outputs/nature_content --section-id Sec71`
- 针对已检索 JSONL 批量抓取：
  ```bash
  python nature_all_in_one.py postfetch --require-cc-by --jsonl outputs/search_run/articles.jsonl --out outputs/nature_content --workers 6 --max-figs 12 --sort year_desc
  ```

## 输出目录结构
```
<out>/<article-id>/
  figures/
    fig_001.jpg
    fig_001.txt
  source_data/
    Source_Data_Fig_4.xlsx
  meta/
    figures.json
    source_data.json
    _source_data_manifest.json
    provenance.json
```
- 仅当成功抓到至少一张图时才保留该文章目录，并同步下载 Source data。
- Source data 文件必须枚举为 `supplementary_information` 或 `reconstructed`；reconstructed 默认 `unverified`，标成 verified 时必须给验证证据。
- `_processed.txt` 记录已完成文章，重复运行时会自动跳过，可通过 `--processed-file` 指向新文件来重新抓取。

## 合规与性能提示
- 所有请求均带自定义 User-Agent，并默认限速（`--sleep` 控制）；大批量时务必设置 `--mailto` 以便 Crossref 识别。
- `discover` smoke 仅访问 Crossref，不会抓图、caption、正文或付费附件。
- `--max-empty-figs` 用于限制连续空页次数（默认 2），可大幅减少无图文章的探测开销。
- 大文件下载使用 300s 网络超时及流式写入，确保 Source data 稳定抓取。
- `outputs/` 目录已加入 `.gitignore`，运行结果不会被推送。

## 版本与运行记录
- 版本信息请查看 `VERSION_LOG.md`，执行日志见 `RUN_LOG.md`。