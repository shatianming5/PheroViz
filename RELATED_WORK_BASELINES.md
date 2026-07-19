# Related Work / Baselines (2024–2026) for PheroViz C2

Scope: recent work relevant to **PheroViz C2** — agentic, multi-round Matplotlib
visualization generation with a **Judge++** that scores *data fidelity* (rendered
Figure/Axes/Artists checked programmatically against the ground-truth source
table), benchmarked on **real published multi-panel scientific figures** (Nature
Communications / eLife / EMBO) that ship per-figure Source Data.

Buckets: **(1)** agentic/iterative viz generation · **(2)** chart-to-code /
figure-to-code · **(3)** NL2VIS generation + evaluation · **(4)** chart /
scientific-figure understanding benchmarks · **(5)** data-fidelity / VLM-as-judge
evaluation.

> **Verification:** every entry below was independently checked by me against the
> arXiv API (`export.arxiv.org/api/query`) and/or the ACL Anthology page — exact
> title, ID, and venue confirmed. No hallucinated citations. "Data-fidelity" =
> a PheroViz-style source-table→rendered-mark check (few prior methods do this;
> that is C2's gap).

| # | Method | 1st author | Venue / Year | arXiv / DOI | B | Task | Input → Output | Iterative / agentic | Data-fidelity vs source table | Multi-panel / sci |
|--:|---|---|---|---|:-:|---|---|---|---|---|
| 1 | **LIDA** | Dibia | ACL Demo 2023 | 10.18653/v1/2023.acl-demo.11 | 1 | LLM viz/infographic tool | table + goal → grammar-agnostic chart code | Yes (generate→execute→filter) | No | panels N/S · sci: no |
| 2 | **MatPlotAgent** | Z. Yang | Findings of ACL 2024 | 10.18653/v1/2024.findings-acl.701 | 1 | Agentic scientific plotting | query + data → plot code/render | Yes (visual-feedback debugging) | No (GPT-4V judge, not table audit) | panels N/S · sci: yes |
| 3 | **PlotGen** | Goswami | arXiv 2025 | 2502.00988 | 1 | Multi-agent sci plotting | request + data → Python plot | Yes (numeric+lexical+visual feedback) | Partial (Numeric Feedback Agent checks values) | panels N/S · sci: yes |
| 4 | **nvAgent** | Ouyang | ACL 2025 | 10.18653/v1/2025.acl-long.960 | 1 | Multi-table NL2VIS | NL + tables → viz/code | Yes (processor/composer/validator) | No | panels N/S · sci: no |
| 5 | **Text2Vis** | Rahman | EMNLP 2025 | 10.18653/v1/2025.emnlp-main.1622 | 1/3 | Text→multimodal viz benchmark | table + NL → answer + chart code | Yes (cross-modal actor–critic) | No (LLM chart-accuracy scorer) | panels N/S · sci: no |
| 6 | **PlotCraft** | J. Zhang | arXiv 2025 | 2511.00010 | 1 | Complex/interactive viz benchmark + model | task + data → complex chart code | Yes (single- + multi-turn refine) | No | mixed incl. sci |
| 7 | **DataMagic** | Xie | VLDB 2026 | 2606.20388 | 1/5 | Table → data-insight video | raw table + NL → chart/narration video | Yes (Generate-then-Orchestrate agents) | **Partial** (DVSpec binds marks to data fields + provenance) | panels N/S · sci: no |
| 8 | **ChartMimic** | C. Yang | ICLR 2025 | 2406.09961 | 2 | Chart-image→code benchmark | chart image + instr → rendering code | No | No (reference code/render metrics) | sci-paper charts · yes |
| 9 | **Plot2Code** | Wu | arXiv 2024 | 2405.07990 | 2 | Plot-image→code benchmark | plot image → executable code | No | No (pass rate + GPT-4V rating) | matplotlib-gallery sci |
| 10 | **ChartCoder** | X. Zhao | ACL 2025 | 10.18653/v1/2025.acl-long.363 | 2 | Chart-to-code MLLM | chart image → chart code | No (Snippet-of-Thought) | No (restoration/executability) | panels N/S |
| 11 | **CharTide** | Zheng | ACL 2026 (Main) | 2604.22192 | 2 | Data-centric chart-to-code | chart image → code | Train-time inquiry-driven RL | No (atomic-QA consistency, not source-table) | panels N/S |
| 12 | **VisEval** | N. Chen | arXiv 2024 | 2407.00981 | 3 | NL2VIS benchmark/metrics | NL + DB → viz spec | No | No (validity/legality/readability) | panels N/S · sci: no |
| 13 | **nvBench 2.0** | Luo | arXiv 2025 | 2503.12880 | 3 | Ambiguous Text2VIS | ambiguous NL + table → valid viz(s) | Stepwise pref-opt (no agent loop) | No (gold-viz match) | panels N/S · sci: no |
| 14 | **DeepVIS** | Shuai | IEEE VIS 2025 (TVCG) | 2508.01700 · 10.1109/TVCG.2025.3634645 | 3 | Explainable NL2VIS (CoT) | NL + data → viz + reasoning | Interactive adjust (no autonomous loop) | No | panels N/S · sci: no |
| 15 | **RL-Text2Vis** | Rahman | arXiv 2026 | 2601.04582 | 3 | Multi-objective RL Text2VIS | table + NL → answer + chart | Post-hoc RL reward (no test-time loop) | No (text/code/quality reward) | panels N/S · sci: no |
| 16 | **ChartBench** | Z. Xu | arXiv 2023 | 2312.15915 | 4 | Chart reasoning + data reliability | chart image + Q → answer | No | N/A (QA) — but probes value reliability | 42 categories |
| 17 | **CharXiv** | Z. Wang | arXiv 2024 | 2406.18521 | 4 | Realistic chart understanding | arXiv chart + Q → answer | No | N/A (understanding QA) | sci: yes (2,323 arXiv charts) |
| 18 | **SciFIBench** | Roberts | NeurIPS 2024 D&B | 2405.08807 | 4 | Scientific-figure interpretation | figure + caption + Q → answer | No | N/A (interpretation QA) | sci: yes |
| 19 | **MultiChartQA** | Zhu | NAACL 2025 | 2410.14179 | 4 | Multi-chart reasoning | multiple charts + Q → answer | No | N/A (QA) | multi-chart: yes |
| 20 | **ChartX / ChartVLM** | Xia | arXiv 2024 | 2402.12185 | 4 | Chart reasoning benchmark + model | chart + task → answer/reasoning | No | N/A (reasoning) | multi-discipline |
| 21 | **VLM Judges Can Rank but Cannot Score** | D. Kumar | arXiv 2026 | 2604.25235 | 5 | Reliability of VLM-as-judge | visual response + task → calibrated score interval | No | No — but shows VLM judges' scores are widest exactly on **chart** tasks (~70% interval) | charts are 1 of 14 tasks |
| 22 | **VegaChat** | Hostnik | arXiv 2026 | 2601.15385 | 5 | NL2VIS generation + assessment | NL + data → declarative chart; + Spec/Vision scores | No loop | No (spec-similarity + VLM image score, not source-table) | panels N/S · sci: no |

## Why C2 is different (positioning)
- **Buckets 1–3** generate charts but judge them by *visual* similarity, LLM/VLM
  rating, or spec match — **not** by auditing rendered marks against the
  underlying data table. Closest to fidelity: PlotGen's Numeric Feedback Agent
  (#3) and DataMagic's DVSpec data-binding (#7), but neither does a
  Figure/Axes/Artists-vs-source-table check on real published multi-panel figures.
- **Buckets 4** are *understanding/QA* benchmarks (chart image → answer), the
  inverse of generation.
- **Bucket 5** is the most direct methodological neighbor: VegaChat (#22) proposes
  generation-plus-assessment metrics, and #21 shows VLM judges are *least reliable
  on charts* — motivating C2's programmatic (non-VLM) data-fidelity judge.
- **C2's gap**: agentic multi-round generation **+** programmatic data-fidelity
  scoring against real per-figure Source Data on **multi-panel** scientific figures.

## Suggested baseline / comparison set for experiments
- Generation baselines: **MatPlotAgent (#2)**, **PlotGen (#3)**, **nvAgent (#4)** (agentic viz gen).
- Chart-to-code baselines: **ChartMimic (#8)**, **ChartCoder (#10)**, **CharTide (#11)**.
- Evaluation baselines: **VisEval (#12)** metrics, **VegaChat (#22)** Spec/Vision Score, and #21 as the VLM-judge-reliability reference.

## Verified BibTeX
```bibtex
@inproceedings{dibia2023lida,
  author={Victor Dibia},
  title={{LIDA}: A Tool for Automatic Generation of Grammar-Agnostic Visualizations and Infographics using Large Language Models},
  booktitle={Proc. ACL 2023, System Demonstrations}, year={2023}, doi={10.18653/v1/2023.acl-demo.11}}
@inproceedings{yang2024matplotagent,
  author={Zhiyu Yang and Zihan Zhou and Shuo Wang and Xin Cong and Xu Han and Yukun Yan and Zhenghao Liu and Zhixing Tan and Pengyuan Liu and Dong Yu and Zhiyuan Liu and Xiaodong Shi and Maosong Sun},
  title={{MatPlotAgent}: Method and Evaluation for {LLM}-Based Agentic Scientific Data Visualization},
  booktitle={Findings of ACL 2024}, year={2024}, doi={10.18653/v1/2024.findings-acl.701}}
@misc{goswami2025plotgen,
  author={Kanika Goswami and Puneet Mathur and Ryan A. Rossi and Franck Dernoncourt},
  title={{PlotGen}: Multi-Agent {LLM}-based Scientific Data Visualization via Multimodal Feedback},
  year={2025}, eprint={2502.00988}, archivePrefix={arXiv}}
@inproceedings{ouyang2025nvagent,
  author={Geliang Ouyang and Jingyao Chen and Zhihe Nie and Yi Gui and Yao Wan and Hongyu Zhang and Dongping Chen},
  title={{nvAgent}: Automated Data Visualization from Natural Language via Collaborative Agent Workflow},
  booktitle={Proc. ACL 2025 (Long Papers)}, year={2025}, doi={10.18653/v1/2025.acl-long.960}}
@inproceedings{rahman2025text2vis,
  author={Mizanur Rahman and Md Tahmid Rahman Laskar and Shafiq Joty and Enamul Hoque},
  title={{Text2Vis}: A Challenging and Diverse Benchmark for Generating Multimodal Visualizations from Text},
  booktitle={Proc. EMNLP 2025}, year={2025}, doi={10.18653/v1/2025.emnlp-main.1622}}
@misc{zhang2025plotcraft,
  author={Jiajun Zhang and Jianke Zhang and Zeyu Cui and Jiaxi Yang and Lei Zhang and Binyuan Hui and Qiang Liu and Zilei Wang and Liang Wang and Junyang Lin},
  title={{PlotCraft}: Pushing the Limits of {LLM}s for Complex and Interactive Data Visualization},
  year={2025}, eprint={2511.00010}, archivePrefix={arXiv}}
@misc{xie2026datamagic,
  author={Yupeng Xie and Chen Ma and Zhenyang Wang and Liangwei Wang and Jiayi Zhu and Chuxuan Zeng and Zhouan Shen and Boyan Li and Yuyu Luo},
  title={{DataMagic}: Transforming Tabular Data into Data Insight Video},
  year={2026}, eprint={2606.20388}, archivePrefix={arXiv}, note={VLDB 2026}}
@inproceedings{yang2025chartmimic,
  author={Cheng Yang and Chufan Shi and Yaxin Liu and Bo Shui and Junjie Wang and Mohan Jing and Linran Xu and Xinyu Zhu and Siheng Li and Yuxiang Zhang and Gongye Liu and Xiaomei Nie and Deng Cai and Yujiu Yang},
  title={{ChartMimic}: Evaluating {LMM}'s Cross-Modal Reasoning Capability via Chart-to-Code Generation},
  booktitle={Proc. ICLR 2025}, year={2025}, eprint={2406.09961}, archivePrefix={arXiv}}
@misc{wu2024plot2code,
  author={Chengyue Wu and Yixiao Ge and Qiushan Guo and Jiahao Wang and Zhixuan Liang and Zeyu Lu and Ying Shan and Ping Luo},
  title={{Plot2Code}: A Comprehensive Benchmark for Evaluating Multi-modal Large Language Models in Code Generation from Scientific Plots},
  year={2024}, eprint={2405.07990}, archivePrefix={arXiv}}
@inproceedings{zhao2025chartcoder,
  author={Xuanle Zhao and Xianzhen Luo and Qi Shi and Chi Chen and Shuo Wang and Zhiyuan Liu and Maosong Sun},
  title={{ChartCoder}: Advancing Multimodal Large Language Model for Chart-to-Code Generation},
  booktitle={Proc. ACL 2025 (Long Papers)}, year={2025}, doi={10.18653/v1/2025.acl-long.363}}
@misc{zheng2026chartide,
  author={Xiangxi Zheng and Kuang He and Jiayi Hu and Ping Yu and Rui Yan and Yuan Yao and Peng Hou and Anxiang Zeng and Alex Jinpeng Wang},
  title={{CharTide}: Data-Centric Chart-to-Code Generation via Tri-Perspective Tuning and Inquiry-Driven Evolution},
  year={2026}, eprint={2604.22192}, archivePrefix={arXiv}, note={ACL 2026 Main}}
@misc{chen2024viseval,
  author={Nan Chen and Yuge Zhang and Jiahang Xu and Kan Ren and Yuqing Yang},
  title={{VisEval}: A Benchmark for Data Visualization in the Era of Large Language Models},
  year={2024}, eprint={2407.00981}, archivePrefix={arXiv}}
@misc{luo2025nvbench2,
  author={Tianqi Luo and Chuhan Huang and Leixian Shen and Boyan Li and Shuyu Shen and Wei Zeng and Nan Tang and Yuyu Luo},
  title={{nvBench} 2.0: Resolving Ambiguity in Text-to-Visualization through Stepwise Reasoning},
  year={2025}, eprint={2503.12880}, archivePrefix={arXiv}}
@article{shuai2025deepvis,
  author={Zhihao Shuai and Boyan Li and Siyu Yan and Yuyu Luo and Weikai Yang},
  title={{DeepVIS}: Bridging Natural Language and Data Visualization Through Step-wise Reasoning},
  journal={IEEE Transactions on Visualization and Computer Graphics}, year={2025},
  doi={10.1109/TVCG.2025.3634645}, note={IEEE VIS 2025}}
@misc{rahman2026rltext2vis,
  author={Mizanur Rahman and Mohammed Saidul Islam and Md Tahmid Rahman Laskar and Shafiq Joty and Enamul Hoque},
  title={Aligning Text, Code, and Vision: A Multi-Objective Reinforcement Learning Framework for Text-to-Visualization},
  year={2026}, eprint={2601.04582}, archivePrefix={arXiv}}
@misc{xu2023chartbench,
  author={Zhengzhuo Xu and Sinan Du and Yiyan Qi and Chengjin Xu and Chun Yuan and Jian Guo},
  title={{ChartBench}: A Benchmark for Complex Visual Reasoning in Charts},
  year={2023}, eprint={2312.15915}, archivePrefix={arXiv}}
@misc{wang2024charxiv,
  author={Zirui Wang and Mengzhou Xia and Luxi He and Howard Chen and Yitao Liu and Richard Zhu and Kaiqu Liang and Xindi Wu and Haotian Liu and Sadhika Malladi and Alexis Chevalier and Sanjeev Arora and Danqi Chen},
  title={{CharXiv}: Charting Gaps in Realistic Chart Understanding in Multimodal {LLM}s},
  year={2024}, eprint={2406.18521}, archivePrefix={arXiv}}
@inproceedings{roberts2024scifibench,
  author={Jonathan Roberts and Kai Han and Neil Houlsby and Samuel Albanie},
  title={{SciFIBench}: Benchmarking Large Multimodal Models for Scientific Figure Interpretation},
  booktitle={NeurIPS 2024 Datasets and Benchmarks}, year={2024}, eprint={2405.08807}, archivePrefix={arXiv}}
@inproceedings{zhu2025multichartqa,
  author={Zifeng Zhu and Mengzhao Jia and Zhihan Zhang and Lang Li and Meng Jiang},
  title={{MultiChartQA}: Benchmarking Vision-Language Models on Multi-Chart Problems},
  booktitle={Proc. NAACL 2025}, year={2025}, eprint={2410.14179}, archivePrefix={arXiv}}
@misc{xia2024chartx,
  author={Renqiu Xia and Bo Zhang and Hancheng Ye and Xiangchao Yan and Qi Liu and Hongbin Zhou and Zijun Chen and Peng Ye and Min Dou and Botian Shi and Junchi Yan and Yu Qiao},
  title={{ChartX} \& {ChartVLM}: A Versatile Benchmark and Foundation Model for Complicated Chart Reasoning},
  year={2024}, eprint={2402.12185}, archivePrefix={arXiv}}
@misc{kumar2026vlmjudges,
  author={Divake Kumar and Sina Tayebati and Devashri Naik and Ranganath Krishnan and Amit Ranjan Trivedi},
  title={{VLM} Judges Can Rank but Cannot Score: Task-Dependent Uncertainty in Multimodal Evaluation},
  year={2026}, eprint={2604.25235}, archivePrefix={arXiv}}
@misc{hostnik2026vegachat,
  author={Marko Hostnik and Rauf Kurbanov and Yaroslav Sokolov and Artem Trofimov},
  title={{VegaChat}: A Robust Framework for {LLM}-Based Chart Generation and Assessment},
  year={2026}, eprint={2601.15385}, archivePrefix={arXiv}}
```
