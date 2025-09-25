Nature ç³»åˆ—æ£€ç´¢ä¸Žå†…å®¹æŠ“å–ï¼ˆåˆè§„ã€å¯æŽˆæƒæ‰©å±•ï¼‰

é¡¹ç›®ç®€ä»‹
- åŸºäºŽå®˜æ–¹/å…è®¸çš„ API å®Œæˆå¯¹ Nature å®¶æ—æœŸåˆŠçš„æ£€ç´¢ä¸Žæ‘˜è¦æ±‡æ€»ï¼ˆCrossref + Europe PMC + PMCï¼‰ã€‚
- åœ¨ä½ å·²ç¡®è®¤æ‹¥æœ‰æŽˆæƒçš„å‰æä¸‹ï¼Œå¯ä»Ž nature.com æŠ“å–å›¾åƒä¸Ž captionã€Source data é™„ä»¶ï¼ˆä¸åšä»»ä½•ç»•è¿‡ï¼Œæ˜Žç¡® UAã€é™é€Ÿä¸Žé‡è¯•ï¼‰ã€‚

èƒ½åŠ›èŒƒå›´
- æ£€ç´¢ï¼šCrossref æœç´¢å¹¶ç­›é€‰â€œNature å®¶æ—â€æœŸåˆŠæ¡ç›®ï¼›Europe PMC è¡¥å…¨æ‘˜è¦ã€PMC/PMID ä¿¡æ¯ï¼›å¯¼å‡º CSV/JSONLã€‚
- PMC å›¾åƒï¼šä»…å½“å­˜åœ¨ PMC å¼€æ”¾èŽ·å–ç‰ˆæœ¬æ—¶ï¼Œä»Ž pmc.ncbi.nlm.nih.gov æŠ“å–å›¾åƒï¼ˆåˆè§„ä½¿ç”¨ï¼‰ã€‚
- æŽˆæƒæŠ“å–ï¼ˆéœ€è¦ä½ ç¡®è®¤æœ‰æƒä½¿ç”¨ï¼‰ï¼š
  - å›¾åƒ + captionï¼šä»Ž nature.com çš„å›¾é¡µæŠ“å–å›¾ç‰‡ä¸Žè¯´æ˜Žã€‚
  - Source dataï¼šä»Ž nature.com æ–‡ç« é¡µæŠ“å–â€œSource dataâ€é™„ä»¶ï¼ˆå« Fig. xã€Extended Data Fig. xï¼‰ã€‚
- ç»„ç»‡åŒ–è¾“å‡ºï¼šæ‰€æœ‰æŠ“å–å†…å®¹æŒ‰æ–‡ç« åˆ†ç›®å½•ï¼Œç»“æž„æ¸…æ™°ã€ä¾¿äºŽåŽå¤„ç†ã€‚

å¿«é€Ÿå¼€å§‹
- æŽ¨è Python 3.9+ï¼›è„šæœ¬ä¼šæŒ‰éœ€è‡ªåŠ¨å®‰è£… `requests`ã€`beautifulsoup4`ã€‚
- åŸºç¡€æ£€ç´¢ï¼ˆåˆè§„ã€æ— ç»•è¿‡ï¼‰ï¼š
  - `python scripts/nature_cli.py --query "cancer" --max 5 --images --out outputs/search_run`
  - å¯é€‰å‚æ•°ï¼š
    - `--timeout 30`ã€`--max-retries 3` ç¨³å¥é‡è¯•
    - `--append` è¿½åŠ å†™å…¥ JSONLï¼ˆæŒ‰ DOI åŽ»é‡ï¼‰
    - `--no-family-bias` å–æ¶ˆå¯¹ `container-title=Nature` çš„åç½®ï¼ˆæ‰©å¤§èŒƒå›´ï¼‰
    - `--mailto you@example.com` ä¸º Crossref æä¾›é‚®ç®±æ ‡è¯†

全功能入口（all-in-one，推荐）
- 本仓库已将功能整合为单一脚本：`nature_all_in_one.py`
- 子命令与示例：
  - 自动搜索+抓取全部（图像+caption、Source data）：
    - 两阶段（先搜索后抓取）：
      - `python nature_all_in_one.py auto --max-per-keyword 50 --max-articles 200 --max-figs 12 --sort year_desc`
    - 流式模式（边搜索边抓取，发现一篇立即抓取）：
      - `python nature_all_in_one.py auto --stream --max-per-keyword 50 --max-articles 200 --max-figs 12`
      - `--stream-workers 6` 支持流式抓取阶段并行下载，Rich 实时展示关键词进度与各 worker 状态（默认 1）
    - 可选：`--keywords-file keywords.txt`（每行一个关键词）、`--mailto you@example.com`、`--sleep 1.0`、`--timeout 30`、`--max-retries 3`
    - 说明：内置多领域关键词已扩展至约 150+（含中英文）；如需自定义请使用 `--keywords-file`
  - 仅检索：
    - `python nature_all_in_one.py search --query "cancer" --max 20 --out outputs/search_run --append`
  - 仅抓取图像页（需授权）：
    - `python nature_all_in_one.py fig --url "https://www.nature.com/articles/<article-id>/figures/1" --out outputs/nature_content`
  - 仅抓取 Source data（需授权）：
    - `python nature_all_in_one.py source --url "https://www.nature.com/articles/<article-id>" --out outputs/nature_content --section-id Sec71 --filter "Fig. 4"`
  - 对检索结果批量抓取（始终“抓取全部”）：
    - `python nature_all_in_one.py postfetch --jsonl outputs/search_run/articles.jsonl --out outputs/nature_content --max-figs 12 --max-articles 200 --sort year_desc`

ç»Ÿä¸€è¾“å‡ºç»“æž„ï¼ˆç›´è§‚ã€ä¾¿äºŽå¤„ç†ï¼‰
- æ‰€æœ‰å†…å®¹æŒ‰æ–‡ç«  ID å­˜æ”¾ï¼š`<out>/<article-id>/`
  - `figures/` å­˜å›¾ä¸ŽåŒå caption æ–‡æœ¬ï¼š
    - `fig_001.jpg`ï¼ˆæˆ– .png ç­‰ï¼‰
    - `fig_001.txt`ï¼ˆå¯¹åº” captionï¼‰
  - `source_data/` å­˜æ”¾æºæ•°æ®é™„ä»¶ï¼š
    - `Source_Data_Fig_4.xlsx`ï¼ˆç¤ºä¾‹ï¼›åŸºäºŽé“¾æŽ¥æ ‡ç­¾æ¸…æ´—å¹¶ä¿ç•™æ‰©å±•åï¼‰
  - `meta/` å…ƒæ•°æ®ä¸Žæ¸…å•ï¼š
    - `figures.json`ï¼šæ¯å¼ å›¾çš„ image_urlã€caption_fileã€image_fileã€figure_no ç­‰
    - `source_data.json`ï¼šæ¯ä¸ªé™„ä»¶çš„ labelã€urlã€saved_asã€orig_name
    - `_source_data_manifest.json`ï¼šæœ¬æ¬¡å‘çŽ°çš„æºæ•°æ®é“¾æŽ¥æ¸…å•

åˆè§„è¯´æ˜Ž
- åŸºç¡€æ£€ç´¢ä½¿ç”¨ Crossref ä¸Ž Europe PMC/PMC çš„å…¬å¼€ APIï¼Œè®¾ç½®æ˜Žç¡® User-Agent å’Œé™é€Ÿï¼ˆé»˜è®¤ 1 req/sï¼‰ã€‚
- æŽˆæƒæŠ“å–è„šæœ¬ä»…åœ¨ä½ ç¡®è®¤æ‹¥æœ‰ç›¸åº”æƒé™æ—¶ä½¿ç”¨ï¼Œä¸åšä»»ä½•åçˆ¬ç»•è¿‡ï¼›å»ºè®®ä¿ç•™ UA ä¸Žåˆç†é™é€Ÿï¼Œé¿å…å¯¹ç«™ç‚¹é€ æˆè´Ÿæ‹…ã€‚
- å†åˆ©ç”¨æ—¶è¯·éµå®ˆç‰ˆæƒä¸Žè®¸å¯æ¡æ¬¾ï¼ˆå¼€æ”¾èŽ·å–ä¹Ÿå¯èƒ½å¯¹å›¾ç‰‡å†ä½¿ç”¨æœ‰é™åˆ¶ï¼‰ã€‚

Git ä¸Žè®°å½•
- æ¯æ­¥ä¿®æ”¹å‡æäº¤æŽ¨é€ï¼›ç‰ˆæœ¬/å˜æ›´è®°å½•è§ `VERSION_LOG.md`ï¼Œè¿è¡Œè®°å½•è§ `RUN_LOG.md`ã€‚


- 加速过滤：通过 --max-empty-figs 指定连续多少空图页即提前停止（默认 2），避免无图文章逐页探测到很大编号。
