# RUN 执行日志

说明：记录每次执行命令的参数、环境与结果概要（数据保存在 `outputs/`，该目录被 `.gitignore` 忽略）。

## 2025-09-24 运行1
- 命令：python scripts/env_bootstrap.py --query "cancer" --max 10 --images --out outputs/run_cancer_1 --sleep 1.0
- 环境：Python 3.11.11；Conda 检测（active env: None）
- 结果：成功，保存 10 条记录；如有 PMC 开放获取文章，图像保存至 outputs/run_cancer_1/images/

## 2025-09-24 运行2
- 命令：python scripts/env_bootstrap.py --query "cancer immunotherapy" --max 15 --images --out outputs/run_ci_1 --sleep 1.0
- 环境：Python 3.11.11；Conda 检测（active env: None）
- 结果：成功，保存 15 条记录；如有 PMC 开放获取文章，图像保存至 outputs/run_ci_1/images/

## 2025-09-24 运行3
- 命令：python scripts/env_bootstrap.py --query "肿瘤免疫" --max 15 --images --out outputs/run_cn_imm_1 --sleep 1.0
- 环境：Python 3.11.11；Conda 检测（active env: None）
- 结果：成功，保存 14 条记录；如有 PMC 开放获取文章，图像保存至 outputs/run_cn_imm_1/images/

## 2025-09-24 运行4
- 命令：python scripts/env_bootstrap.py --query "checkpoint inhibitor" --max 20 --images --out outputs/run_ci_checkpt_1 --sleep 1.0 --timeout 30 --max-retries 3 --append
- 环境：Python 3.11.11；Conda 检测（active env: None）
- 结果：成功，保存 20 条记录；如有 PMC 开放获取文章，图像保存至 outputs/run_ci_checkpt_1/images/

## 2025-09-24 运行5
- 命令：python scripts/env_bootstrap.py --query "单细胞 癌症" --max 20 --images --out outputs/run_cn_scRNA_1 --sleep 1.0 --timeout 30 --max-retries 3 --append
- 环境：Python 3.11.11；Conda 检测（active env: None）
- 结果：成功，保存 14 条记录；如有 PMC 开放获取文章，图像保存至 outputs/run_cn_scRNA_1/images/
