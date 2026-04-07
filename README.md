# Long Short-Term Imputer: Handling Consecutive Missing Values in Time Series

> A novel deep learning framework for imputing consecutive missing values in time series data, combining long-term and short-term dependencies with adaptive meta-weighting.

## Authors

**Anonymous authors**

*Paper under double-blind review at TMLR*

## Links

- **Paper**: *Coming soon* (under double-blind review at TMLR)
- **Base Repository**: [Time-Series-Library](https://github.com/thuml/Time-Series-Library)

> 本项目基于 [Time-Series-Library](https://github.com/thuml/Time-Series-Library) 扩展，实现了 LSTI 方法。

---

## Table of Contents

- [Updates](#updates)
- [Introduction](#introduction)
- [Highlights](#highlights)
- [Method / Framework](#method--framework)
- [Project Structure](#project-structure)
- [Installation](#installation)
- [Checkpoints / Models](#checkpoints--models)
- [Dataset / Benchmark](#dataset--benchmark)
- [Usage](#usage)
- [Results](#results)
- [TODO](#todo)
- [Citation](#citation)
- [Acknowledgement](#acknowledgement)
- [License](#license)

---

## Updates

- [04/2026] Initial release of LSTI implementation
- [04/2026] Paper submitted to TMLR

---

## Introduction

本项目是论文 **Long Short-Term Imputer: Handling Consecutive Missing Values in Time Series (LSTI)** 的官方实现。

### Problem Statement

时间序列数据中频繁出现的缺失值会显著阻碍时间序列分析。现有的深度学习插补方法主要针对"完全随机缺失"(MCAR)场景设计，但现实世界中的缺失值往往由于信号丢失、环境干扰和设备故障等原因连续出现并成簇分布。

特别是"Blackout"模式——所有通道的缺失值在相同位置对齐——是最具挑战性的情况，因为标准插补方法无法利用其他通道或时间点的信息来重建缺失值。

### Method Overview

我们提出 **LSTI (Long Short-Term Imputer)** 来处理不同间隔长度的连续缺失值:

- **Long-Term Imputer**: 使用双向自回归思想设计，包含前向预测模型和后向预测模型，通过一致性正则化训练，能够捕获长期时间依赖并适应长间隔连续缺失值

- **Short-Term Imputer**: 设计用于捕获短期时间依赖，可以有效插补短间隔连续缺失值

- **Meta-weighting Network**: 自适应学习长期和短期依赖的重要性，动态平衡两个插补器的输出

### Key Contributions

- 提出 Long-Term Imputer 捕获时间序列的长期依赖，使用前向和后向预测网络双向自回归插补缺失值
- 提出 Short-Term Imputer 捕获短期依赖，配合 Meta-weighting 模块自适应平衡长短期依赖权重
- 在五个真实世界数据集上进行了大量实验，相比当前最先进的深度学习插补方法平均降低57.4%的误差

---

## Highlights

- 提出 **LSTI (Long Short-Term Imputer)**，专注于时间序列 **连续缺失值插补 (Imputation)** 任务
- Long-Term Imputer 使用双向自回归 + 一致性正则化，有效处理长间隔连续缺失值
- Short-Term Imputer + Meta-weighting Network 自适应平衡长短期依赖
- 在五个真实世界数据集（Electricity, Traffic, METR-LA, Guangzhou, PEMS04）上平均降低 **57.4%** 误差
- 基于 [Time-Series-Library](https://github.com/thuml/Time-Series-Library) 扩展，同时支持预测、异常检测、分类等任务

---

## Method / Framework

LSTI 框架包含三个主要组件:

1. **Long-Term Imputer**: 双向自回归网络，前向和后向预测网络分别从两个方向自回归地插补整个序列，通过一致性正则化训练
2. **Short-Term Imputer**: 自映射网络，使用随机生成的连续缺失掩码训练，捕获短期依赖
3. **Meta-weighting Module**: 学习当前数据中长期和短期依赖的比例，自适应平衡权重

> 方法框架图请参阅论文原文。

---

## Project Structure

```text
.
├── data_provider/              # 数据加载和处理
├── exp/                        # 实验主代码 (训练、测试逻辑)
│   ├── exp_imputation.py       # 通用插补实验
│   ├── exp_imputation_my.py    # 扩展插补实验
│   ├── exp_LSTI_onlyLong.py    # LSTI Long-Term Imputer 实验专有
│   ├── exp_forcastImputation_AR.py
│   ├── exp_forcastImputation_3M_auto.py
│   └── ...
├── layers/                     # 网络层组件
├── models/                     # 模型实现
│   ├── TimesNet_AR.py          # TimesNet 自回归版本 (LSTI backbone)
│   ├── TimesNet.py             # TimesNet 原始版本
│   ├── iTransformer_AR.py
│   ├── Transformer_AR.py
│   ├── Transformer_GPT.py
│   └── ...
├── scripts/                    # 实验脚本
│   ├── imputation/             # 通用插补脚本
│   ├── forecaseImputation/     # LSTI 专属插补+预测实验脚本
│   └── ...
├── pic/                        # 图片资源
├── utils/                      # 工具函数
├── run.py                      # 原始运行脚本 (Time-Series-Library)
├── myrun.py                    # LSTI 扩展运行脚本 (推荐)
├── requirements.txt
└── README.md
```

---

## Installation

### 1. Clone the repository

```bash
git clone https://github.com/iLearn-Lab/Time-Series-Library_ARGPT.git
cd Time-Series-Library_ARGPT
```

### 2. Create environment

```bash
python -m venv .venv
source .venv/bin/activate   # Linux / Mac
# .venv\Scripts\activate    # Windows
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

---

## Checkpoints / Models

训练好的模型权重会保存在 `./checkpoints/` 目录下。如需使用已有权重，请下载后放入该目录，或在运行脚本时通过 `--checkpoints` 参数指定路径。

---

## Dataset / Benchmark

本仓库支持两类数据集：

### LSTI 论文实验数据集

论文使用以下五个真实世界数据集进行连续缺失值插补实验：

| Dataset | Description | Source |
|---------|-------------|--------|
| Electricity | 321 维电力负荷数据（小时级，2012-2014） | UCI |
| Traffic | 862 维高速公路交通占有率数据（小时级，2015-2016） | Caltrans PeMS |
| METR-LA | 207 维洛杉矶县高速公路交通速度数据 | [Li et al., 2018] |
| Guangzhou | 214 维广州市道路交通速度数据（10分钟级，2016） | [Chen et al., 2018] |
| PEMS04 | 307 维旧金山湾区交通传感器数据（59天，2018） | [Chen et al., 2001] |

> 请将数据放入 `./dataset/` 目录。具体的数据预处理方式请参阅论文 Section 4.1。

### 通用数据集（Time-Series-Library 继承）

以下数据集可用于预测、异常检测、分类等任务：

- **Google Drive**: [Download Link](https://drive.google.com/drive/folders/13Cg1KYOlzM5C7K8gK8NfC-F3EYxkM3D2?usp=sharing)
- **Baidu Drive**: [Download Link](https://pan.baidu.com/s/1r3KhGd0Q9PJIUZdfEYoymg?pwd=i9iy)

| Dataset | Variables | Granularity | Task |
|---------|-----------|-------------|------|
| ETTh1/ETTh2 | 7 | Hourly | Forecasting, Imputation |
| ETTm1/ETTm2 | 7 | 15-min | Forecasting, Imputation |
| Weather | 21 | 10-min | Forecasting, Imputation |
| Electricity | 321 | Hourly | Forecasting, Imputation |
| Traffic | 862 | Hourly | Forecasting, Imputation |

---

## Usage

本项目提供两个入口脚本：

- **`myrun.py`** — LSTI 扩展脚本（推荐），支持 LSTI 专属任务类型
- **`run.py`** — 原始 Time-Series-Library 脚本，支持通用任务

### Imputation（LSTI 核心任务）

使用 `myrun.py` 运行 LSTI 插补实验：

```bash
python -u myrun.py \
  --task_name imputation \
  --is_training 1 \
  --root_path ./dataset/electricity/ \
  --data_path electricity.csv \
  --model_id electricity \
  --model TimesNet \
  --data MyData \
  --features M \
  --seq_len 96 \
  --label_len 0 \
  --pred_len 96 \
  --enc_in 321 \
  --dec_in 321 \
  --c_out 321 \
  --batch_size 16 \
  --d_model 256 \
  --d_ff 256 \
  --des 'Exp' \
  --mask_rate 0.25 \
  --missing_type Blackout
```

或使用预设脚本：

```bash
bash ./scripts/forecaseImputation/exe.sh
```

### LSTI 专属任务类型

`myrun.py` 支持以下 LSTI 专属任务（`run.py` 不支持）：

| Task Name | 说明 |
|-----------|------|
| `forecastImputation` | LSTI 插补+预测联合任务 |
| `forecastImputation_3M_auto` | LSTI 3M 自动模式 |
| `LSTI_onlyLong` | 仅使用 Long-Term Imputer（消融实验） |

### 通用任务（继承自 Time-Series-Library）

```bash
# 使用 run.py 运行通用任务
# 长期预测
bash ./scripts/long_term_forecast/ETT_script/TimesNet_ETTh1.sh
# 短期预测
bash ./scripts/short_term_forecast/TimesNet_M4.sh
# 异常检测
bash ./scripts/anomaly_detection/PSM/TimesNet.sh
# 分类
bash ./scripts/classification/TimesNet.sh
```

### Available Models

项目支持以下模型:

- [x] **TimesNet** / **TimesNet_AR** - TimesNet: Temporal 2D-Variation Modeling for General Time Series Analysis [[ICLR 2023]](https://openreview.net/pdf?id=ju_Uqw384Oq)
- [x] **iTransformer** / **iTransformer_AR** - iTransformer: Inverted Transformers Are Effective for Time Series Forecasting [[ICLR 2024]](https://arxiv.org/abs/2310.06625)
- [x] **Transformer** / **Transformer_AR** - Attention is All You Need [[NeurIPS 2017]]
- [x] **Autoformer** - Autoformer: Decomposition Transformers with Auto-Correlation [[NeurIPS 2021]]
- [x] **DLinear** - Are Transformers Effective for Time Series Forecasting? [[AAAI 2023]]
- [x] **PatchTST** - A Time Series is Worth 64 Words [[ICLR 2023]]
- [x] **Informer** - Beyond Efficient Transformer for Long Sequence [[AAAI 2021]]
- [x] **FEDformer**, **ETSformer**, **Crossformer**, **Koopa**, **FreTS**, **TiDE**, **FiLM**, **MICN**, **LightTS**, **Pyraformer**, **Reformer**, **Non-stationary Transformer**

---

## Results

在五个真实世界数据集上，LSTI 相比当前最先进的深度学习插补方法平均降低 **57.4%** 的误差。各数据集上的 MSE 改进幅度分别为：Electricity 72.32%, Traffic 55.43%, METR-LA 40.27%, Guangzhou 54.54%, PEMS04 68.29%。

> 详细实验结果和对比方法请参阅论文 Table 1-5。

---

## TODO

- [ ] 补充方法框架图（论文 Figure 2）到 `./pic/` 目录
- [ ] 补充 LSTI 论文实验数据集的下载说明
- [ ] 上传预训练模型权重（如 `LSTI_onlyLong_guangzhou_TimesNet_AR`）
- [ ] 补充 `forecastImputation` 等专属任务的详细参数说明

---

## Citation

如果您觉得本项目有用，请引用我们的论文:

```bibtex
@misc{lsti2026,
  title={Long Short-Term Imputer: Handling Consecutive Missing Values in Time Series},
  author={Anonymous Authors},
  year={2026},
  note={Under review at TMLR}
}
```

同时也请引用原始 Time-Series-Library:

```bibtex
@inproceedings{wu2023timesnet,
  title={TimesNet: Temporal 2D-Variation Modeling for General Time Series Analysis},
  author={Haixu Wu and Tengge Hu and Yong Liu and Hang Zhou and Jianmin Wang and Mingsheng Long},
  booktitle={International Conference on Learning Representations},
  year={2023},
}
```

---

## Acknowledgement

本项目基于以下开源项目构建:

- [Time-Series-Library](https://github.com/thuml/Time-Series-Library) - 基础框架和模型实现
- [Autoformer](https://github.com/thuml/Autoformer) - 预测和插补基线
- [Anomaly-Transformer](https://github.com/thuml/Anomaly-Transformer) - 异常检测基线

感谢开源社区提供的有用基线和工具。

---

## License

This project is released under the Apache License 2.0.
