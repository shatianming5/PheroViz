Nature 系列检索与内容抓取（合规、可授权扩展）

项目简介
- 基于官方/允许的 API 完成对 Nature 家族期刊的检索与摘要汇总（Crossref + Europe PMC + PMC）。
- 在你已确认拥有授权的前提下，可从 nature.com 抓取图像与 caption、Source data 附件（不做任何绕过，明确 UA、限速与重试）。

能力范围
- 检索：Crossref 搜索并筛选“Nature 家族”期刊条目；Europe PMC 补全摘要、PMC/PMID 信息；导出 CSV/JSONL。
- PMC 图像：仅当存在 PMC 开放获取版本时，从 pmc.ncbi.nlm.nih.gov 抓取图像（合规使用）。
- 授权抓取（需要你确认有权使用）：
  - 图像 + caption：从 nature.com 的图页抓取图片与说明。
  - Source data：从 nature.com 文章页抓取“Source data”附件（含 Fig. x、Extended Data Fig. x）。
- 组织化输出：所有抓取内容按文章分目录，结构清晰、便于后处理。

快速开始
- 推荐 Python 3.9+；脚本会按需自动安装 `requests`、`beautifulsoup4`。
- 基础检索（合规、无绕过）：
  - `python scripts/nature_cli.py --query "cancer" --max 5 --images --out outputs/search_run`
  - 可选参数：
    - `--timeout 30`、`--max-retries 3` 稳健重试
    - `--append` 追加写入 JSONL（按 DOI 去重）
    - `--no-family-bias` 取消对 `container-title=Nature` 的偏置（扩大范围）
    - `--mailto you@example.com` 为 Crossref 提供邮箱标识

单文件入口（all‑in‑one，推荐）
- 本仓库已将功能整合为单一脚本：`nature_all_in_one.py`
- 子命令与示例：
  - 自动搜索+抓取全部（图像+caption、Source data）：
    - 两阶段（先搜后抓）：
      - `python nature_all_in_one.py auto --max-per-keyword 50 --max-articles 200 --max-figs 12 --sort year_desc`
    - 流式（边搜边抓，每发现一篇立即抓取）：
      - `python nature_all_in_one.py auto --stream --max-per-keyword 50 --max-articles 200 --max-figs 12`
    - 可选：`--keywords-file keywords.txt`（每行一个关键词）、`--mailto you@example.com`、`--sleep 1.0`、`--timeout 30`、`--max-retries 3`
    - 说明：内置多领域关键词已扩展至约 150+（含中英文）；如需自定义请使用 `--keywords-file`
  - 仅检索：
    - `python nature_all_in_one.py search --query "cancer" --max 20 --out outputs/search_run --append`
  - 仅抓取某图页（需授权）：
    - `python nature_all_in_one.py fig --url "https://www.nature.com/articles/<article-id>/figures/1" --out outputs/nature_content`
  - 仅抓取某文 Source data（需授权）：
    - `python nature_all_in_one.py source --url "https://www.nature.com/articles/<article-id>" --out outputs/nature_content --section-id Sec71 --filter "Fig. 4"`
  - 对检索结果批量抓取（始终“抓取全部”）：
    - `python nature_all_in_one.py postfetch --jsonl outputs/search_run/articles.jsonl --out outputs/nature_content --max-figs 12 --max-articles 200 --sort year_desc`

统一输出结构（直观、便于处理）
- 所有内容按文章 ID 存放：`<out>/<article-id>/`
  - `figures/` 存图与同名 caption 文本：
    - `fig_001.jpg`（或 .png 等）
    - `fig_001.txt`（对应 caption）
  - `source_data/` 存放源数据附件：
    - `Source_Data_Fig_4.xlsx`（示例；基于链接标签清洗并保留扩展名）
  - `meta/` 元数据与清单：
    - `figures.json`：每张图的 image_url、caption_file、image_file、figure_no 等
    - `source_data.json`：每个附件的 label、url、saved_as、orig_name
    - `_source_data_manifest.json`：本次发现的源数据链接清单

合规说明
- 基础检索使用 Crossref 与 Europe PMC/PMC 的公开 API，设置明确 User-Agent 和限速（默认 1 req/s）。
- 授权抓取脚本仅在你确认拥有相应权限时使用，不做任何反爬绕过；建议保留 UA 与合理限速，避免对站点造成负担。
- 再利用时请遵守版权与许可条款（开放获取也可能对图片再使用有限制）。

Git 与记录
- 每步修改均提交推送；版本/变更记录见 `VERSION_LOG.md`，运行记录见 `RUN_LOG.md`。
