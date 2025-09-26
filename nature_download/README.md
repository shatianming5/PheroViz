# Nature 系列检索与内容抓取（合规、可授权扩展）

## 项目简介
- 基于官方或授权许可的 API，完成对 Nature 家族期刊的检索、摘要汇总与补全（Crossref + Europe PMC + PMC）。
- 在确保已获授权的前提下，可从 nature.com 抓取图像及 caption，以及 Source data 附件；脚本会主动带上合规 UA、限速与重试策略。

## 核心能力
- **检索聚合**：Crossref 搜索并过滤 Nature 家族条目，Europe PMC 补全文摘、PMC/PMID、开放获取状态等信息，输出 JSONL/CSV。
- **图像抓取**：解析 `https://www.nature.com/articles/<article-id>/figures/<n>`，提取原图 URL、caption，自动清理无图页面、连续空页可控。
- **Source data 下载**：解析文章页的 “Source data” 区块，逐个保存附件并生成 manifest，支持大文件流式下载与超时控制。
- **输出组织**：按文章 ID 归档到 `<out>/<article-id>/`，细分 `figures/`、`source_data/`、`meta/`，同时维护 processed 记录用于跳过已完成内容。

## 快速开始
- 建议 Python 3.9+ 环境；脚本会按需安装 `requests`、`beautifulsoup4`、`rich`。
- 基础检索（合规、不抓取）：
  ```bash
  python scripts/nature_cli.py --query "cancer" --max 5 --images --out outputs/search_run
  ```
  常用可选项：`--timeout 30`、`--max-retries 3`、`--append`、`--no-family-bias`、`--mailto you@example.com`。

## 全功能入口 `nature_all_in_one.py`
- 仓库已将全部能力整合在单脚本中，可通过子命令组合使用。
- 自动搜索 + 抓取全部（图像 + caption + Source data）：
  - **两阶段（先搜后抓）**
    ```bash
    python nature_all_in_one.py auto --max-per-keyword 50 --max-articles 200 --max-figs 12 --sort year_desc
    ```
  - **流式模式（边搜边抓）**
    ```bash
    python nature_all_in_one.py auto --stream --stream-workers 6 --max-per-keyword 50 --max-articles 200 --max-figs 12
    ```
    `--stream-workers` 控制抓取线程数（默认 1），Rich 会显示关键词进度、成功数量与各 worker 状态。
  - 其他常用可选项：`--keywords-file keywords.txt`（每行一个关键词）、`--mailto you@example.com`、`--sleep 1.0`、`--timeout 300`、`--max-retries 5`。
- 仅检索：`python nature_all_in_one.py search --query "cancer" --max 20 --out outputs/search_run --append`
- 仅抓图：`python nature_all_in_one.py fig --url "https://www.nature.com/articles/<article-id>/figures/1" --out outputs/nature_content`
- 仅抓 Source data：`python nature_all_in_one.py source --url "https://www.nature.com/articles/<article-id>" --out outputs/nature_content --section-id Sec71`
- 针对已检索 JSONL 批量抓取：
  ```bash
  python nature_all_in_one.py postfetch --jsonl outputs/search_run/articles.jsonl --out outputs/nature_content --workers 6 --max-figs 12 --sort year_desc
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
```
- 仅当成功抓到至少一张图时才保留该文章目录，并同步下载 Source data。
- `_processed.txt` 记录已完成文章，重复运行时会自动跳过，可通过 `--processed-file` 指向新文件来重新抓取。

## 合规与性能提示
- 所有请求均带自定义 User-Agent，并默认限速（`--sleep` 控制）；大批量时务必设置 `--mailto` 以便 Crossref 识别。
- `--max-empty-figs` 用于限制连续空页次数（默认 2），可大幅减少无图文章的探测开销。
- 大文件下载使用 300s 网络超时及流式写入，确保 Source data 稳定抓取。
- `outputs/` 目录已加入 `.gitignore`，运行结果不会被推送。

## 版本与运行记录
- 版本信息请查看 `VERSION_LOG.md`，执行日志见 `RUN_LOG.md`。