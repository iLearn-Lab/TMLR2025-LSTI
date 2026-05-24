# Long Short-Term Imputer: Handling Consecutive Missing Values in Time Series

> A novel deep learning framework for imputing consecutive missing values in time series data, combining long-term and short-term dependencies with adaptive meta-weighting.

## Authors

**Anonymous authors**

*Paper under double-blind review at TMLR*

## Links

- **Paper**: *Coming soon* (under double-blind review at TMLR)
- **Base Repository**: [Time-Series-Library](https://github.com/thuml/Time-Series-Library)

> This project extends [Time-Series-Library](https://github.com/thuml/Time-Series-Library) and implements the LSTI method.

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
- [License](#license)

---

## Updates

- [04/2026] Initial release of LSTI implementation
- [04/2026] Paper submitted to TMLR

---

## Introduction

This repository provides the official implementation of the paper **Long Short-Term Imputer: Handling Consecutive Missing Values in Time Series (LSTI)**.

### Problem Statement

Missing values frequently occurring in time series data can significantly impede time series analysis. Existing deep learning imputation methods are primarily designed for the "Missing Completely At Random" (MCAR) scenario; however, missing values in real-world settings tend to appear consecutively and form clusters due to factors such as signal loss, environmental interference, and device failures.

In particular, the "Blackout" pattern—where missing values across all channels are aligned at the same positions—poses the most challenging scenario, as standard imputation methods are unable to leverage information from other channels or time steps to reconstruct the missing values.

### Method Overview

We propose **LSTI (Long Short-Term Imputer)** to handle consecutive missing values of varying gap lengths:

- **Long-Term Imputer**: Designed with a bidirectional autoregressive scheme, comprising a forward prediction model and a backward prediction model trained with consistency regularization. It captures long-term temporal dependencies and adapts to long-gap consecutive missing values.

- **Short-Term Imputer**: Designed to capture short-term temporal dependencies, effectively imputing short-gap consecutive missing values.

- **Meta-weighting Network**: Adaptively learns the importance of long-term and short-term dependencies, dynamically balancing the outputs of the two imputers.

### Key Contributions

- Propose the Long-Term Imputer to capture long-term dependencies in time series, employing forward and backward prediction networks for bidirectional autoregressive imputation of missing values.
- Propose the Short-Term Imputer to capture short-term dependencies, coupled with a Meta-weighting module to adaptively balance the weights of long-term and short-term dependencies.
- Conduct extensive experiments on five real-world datasets, achieving an average error reduction of 57.4% compared to state-of-the-art deep learning imputation methods.

---

## Highlights

- Propose **LSTI (Long Short-Term Imputer)**, specifically designed for the task of **consecutive missing value imputation** in time series.
- The Long-Term Imputer employs bidirectional autoregression with consistency regularization, effectively handling long-gap consecutive missing values.
- The Short-Term Imputer, combined with the Meta-weighting Network, adaptively balances long-term and short-term dependencies.
- Achieve an average error reduction of **57.4%** across five real-world datasets (Electricity, Traffic, METR-LA, Guangzhou, PEMS04).
- Built upon [Time-Series-Library](https://github.com/thuml/Time-Series-Library), additionally supporting forecasting, anomaly detection, classification, and other tasks.

---

## Method / Framework

The LSTI framework comprises three main components:

1. **Long-Term Imputer**: A bidirectional autoregressive network where the forward and backward prediction networks independently impute the entire sequence in an autoregressive manner from both directions, trained with consistency regularization.
2. **Short-Term Imputer**: A self-mapping network trained with randomly generated consecutive missing masks to capture short-term dependencies.
3. **Meta-weighting Module**: Learns the proportion of long-term versus short-term dependencies in the current data, adaptively balancing their weights.

> Please refer to the original paper for the method framework diagram.

---

## Project Structure

```text
.
├── data_provider/              # Data loading and processing
├── exp/                        # Experiment main code (training & testing logic)
│   ├── exp_imputation.py       # General imputation experiment
│   ├── exp_imputation_my.py    # Extended imputation experiment
│   ├── exp_LSTI_onlyLong.py    # LSTI Long-Term Imputer experiment specific
│   ├── exp_forcastImputation_AR.py
│   ├── exp_forcastImputation_3M_auto.py
│   └── ...
├── layers/                     # Network layer components
├── models/                     # Model implementations
│   ├── TimesNet_AR.py          # TimesNet autoregressive version (LSTI backbone)
│   ├── TimesNet.py             # TimesNet original version
│   ├── iTransformer_AR.py
│   ├── Transformer_AR.py
│   ├── Transformer_GPT.py
│   └── ...
├── scripts/                    # Experiment scripts
│   ├── imputation/             # General imputation scripts
│   ├── forecaseImputation/     # LSTI-specific imputation + forecasting scripts
│   └── ...
├── pic/                        # Image resources
├── utils/                      # Utility functions
├── run.py                      # Original entry script (Time-Series-Library)
├── myrun.py                    # LSTI extended entry script (recommended)
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

Trained model weights are saved in the `./checkpoints/` directory. To use pre-trained weights, download them and place them in this directory, or specify the path via the `--checkpoints` argument when running the scripts.

---

## Dataset / Benchmark

This repository supports two categories of datasets:

### LSTI Paper Experiment Datasets

The paper uses the following five real-world datasets for consecutive missing value imputation experiments:

| Dataset | Description | Source |
|---------|-------------|--------|
| Electricity | 321-dimensional electricity load data (hourly, 2012–2014) | UCI |
| Traffic | 862-dimensional highway traffic occupancy data (hourly, 2015–2016) | Caltrans PeMS |
| METR-LA | 207-dimensional traffic speed data from Los Angeles County highways | [Li et al., 2018] |
| Guangzhou | 214-dimensional urban traffic speed data from Guangzhou (10-min intervals, 2016) | [Chen et al., 2018] |
| PEMS04 | 307-dimensional traffic sensor data from the San Francisco Bay Area (59 days, 2018) | [Chen et al., 2001] |

> Please place the data in the `./dataset/` directory. For detailed data preprocessing procedures, refer to Section 4.1 of the paper.

### General Datasets (Inherited from Time-Series-Library)

The following datasets can be used for forecasting, anomaly detection, classification, and other tasks:

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

This project provides two entry scripts:

- **`myrun.py`** — LSTI extended script (recommended), supporting LSTI-specific task types
- **`run.py`** — Original Time-Series-Library script, supporting general tasks

### Imputation (LSTI Core Task)

Run LSTI imputation experiments using `myrun.py`:

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

Alternatively, use the provided scripts:

```bash
bash ./scripts/forecaseImputation/exe.sh
```

### LSTI-Specific Task Types

`myrun.py` supports the following LSTI-specific tasks (not available in `run.py`):

| Task Name | Description |
|-----------|-------------|
| `forecastImputation` | LSTI joint imputation + forecasting task |
| `forecastImputation_3M_auto` | LSTI 3M automatic mode |
| `LSTI_onlyLong` | Long-Term Imputer only (ablation study) |

### General Tasks (Inherited from Time-Series-Library)

```bash
# Run general tasks using run.py
# Long-term forecasting
bash ./scripts/long_term_forecast/ETT_script/TimesNet_ETTh1.sh
# Short-term forecasting
bash ./scripts/short_term_forecast/TimesNet_M4.sh
# Anomaly detection
bash ./scripts/anomaly_detection/PSM/TimesNet.sh
# Classification
bash ./scripts/classification/TimesNet.sh
```

### Available Models

The project supports the following models:

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

Across five real-world datasets, LSTI achieves an average error reduction of **57.4%** compared to state-of-the-art deep learning imputation methods. The MSE improvements on each dataset are as follows: Electricity 72.32%, Traffic 55.43%, METR-LA 40.27%, Guangzhou 54.54%, PEMS04 68.29%.

> For detailed experimental results and compared methods, please refer to Tables 1–5 in the paper.

---

## TODO

- [ ] Add the method framework diagram (Figure 2 in the paper) to the `./pic/` directory
- [ ] Provide download instructions for the LSTI paper experiment datasets
- [ ] Upload pre-trained model weights (e.g., `LSTI_onlyLong_guangzhou_TimesNet_AR`)
- [ ] Provide detailed parameter descriptions for LSTI-specific tasks such as `forecastImputation`

---

## Citation

If you find this work useful, please cite our paper:

```bibtex
@misc{lsti2026,
  title={Long Short-Term Imputer: Handling Consecutive Missing Values in Time Series},
  author={Anonymous Authors},
  year={2026},
  note={Under review at TMLR}
}
```

---

## License

This project is released under the Apache License 2.0.
