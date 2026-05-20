<h1 align="center">🌊 REPA-P for Physics-Informed Diffusion Models 🚀</h1>

## 🧩 Overview

![REPA-P overview](assets/repa_p_overview.png)

## 📌 Introduction

📚 This repository contains the U-Net implementation used in our REPA-P experiments, built on top of Physics-Informed Diffusion Models (PIDM).

### 🌟 Highlights

- 🔬 **Physics-informed diffusion**: train diffusion models with physics residual or physics-inspired guidance.
- 🧠 **REPA-P**: add projection heads to align intermediate representations with physical constraints.
- 🧱 **PIDM baseline included**: switch between PIDM and REPA-P from the same YAML file.
- 🌀 **Three benchmark systems**: Darcy flow, mechanics, and turbulent flow.
- ⚡ **Simple workflow**: edit `model.yaml`, train with `main.py`, evaluate with `sample.py`.
- 📦 **Release friendly**: no datasets, checkpoints, logs, or DiT files are included.

🧭 Currently supported tasks:

- 🌊 Darcy flow
- 🏗️ Topology optimization / mechanics
- 🌀 Turbulent channel-flow slice

---

## 🚀 Quick Start

```bash
pip install -r requirements.txt
python main.py --config model.yaml --gpu 0
```

`main.py` trains the model. `sample.py` evaluates trained checkpoints. All main settings are controlled in `model.yaml`.

<details>
<summary><b>📁 Data Setup</b></summary>

🚫 Datasets and checkpoints are not included in this GitHub repository.

⬇️ For Darcy flow and mechanics, use the same benchmark data as the original PIDM repository. Download the data and pretrained models from the [ETHZ Research Collection](https://doi.org/10.3929/ethz-b-000674074).

🗂️ After downloading and unzipping, copy the extracted `data/darcy` and `data/mechanics` folders into this repository. The final layout should look like this:

```text
.
├── data
│   ├── darcy
│   │   ├── train
│   │   │   ├── p_data.csv
│   │   │   └── K_data.csv
│   │   └── valid
│   │       ├── p_data.csv
│   │       └── K_data.csv
│   ├── mechanics
│   │   ├── train/fields
│   │   ├── test/valid/fields
│   │   ├── test/test_level_1/fields
│   │   ├── test/test_level_2/fields
│   │   └── solidspy_k_no_BC
│   └── ch_2Dxysec.pickle
└── trained_models
    └── ...                 # optional; only needed for evaluating existing checkpoints
```

💡 If you train from scratch with `main.py`, only the `data/` folder is required. The `trained_models/` folder will be created automatically.

### 🌊 Darcy Flow

The training script expects:

```text
data/darcy/train/p_data.csv
data/darcy/train/K_data.csv
data/darcy/valid/p_data.csv
data/darcy/valid/K_data.csv
```

Set:

```yaml
gov_eqs: darcy
```

### 🏗️ Mechanics

The training script expects:

```text
data/mechanics/train/fields/
data/mechanics/test/valid/fields/
data/mechanics/test/test_level_1/fields/
data/mechanics/test/test_level_2/fields/
data/mechanics/solidspy_k_no_BC/
```

Set:

```yaml
gov_eqs: mechanics
```

### 🌀 Turbulent Flow

Place the processed turbulent channel-flow pickle file here:

```text
data/ch_2Dxysec.pickle
```

Expected array layout:

```text
[T, X, Y, C] = [10000, 128, 48, 1]
```

Set:

```yaml
gov_eqs: turbulent
turbulent_data_path: ./data/ch_2Dxysec.pickle
```

</details>

<details open>
<summary><b>🛠️ Training: PIDM vs REPA-P</b></summary>

The same training script supports both PIDM and REPA-P. Change the switch in `model.yaml`.

### 🧪 PIDM baseline

```yaml
run_name: darcy.pidm.unet
use_projection_heads: False
projection_positions: []
projection_hidden_dim: 0
c_projection: 0.0
```

### 🔥 REPA-P

```yaml
run_name: darcy.repap.unet
use_projection_heads: True
projection_positions:
  - decoder
projection_hidden_dim: 128
c_projection: 0.01
```

💡 If `use_projection_heads: True`, keep `projection_positions` non-empty and set `c_projection > 0`.

Train with:

```bash
python main.py --config model.yaml --gpu 0
```

📦 Training outputs are saved to:

```text
trained_models/<run_name>/
```

</details>

<details>
<summary><b>⚙️ Typical Dataset Settings</b></summary>

Use these typical settings in `model.yaml`.

### 🌊 Darcy

```yaml
gov_eqs: darcy
train_iterations: 150000
train_batch_size: 64
c_ineq: 0.0
lambda_opt: 0.0
```

### 🏗️ Mechanics

```yaml
gov_eqs: mechanics
train_iterations: 300000
train_batch_size: 8
c_ineq: 0.001
lambda_opt: 0.1
```

### 🌀 Turbulent

```yaml
gov_eqs: turbulent
train_iterations: 150000
train_batch_size: 64
turbulent_data_path: ./data/ch_2Dxysec.pickle
```

</details>

<details>
<summary><b>📊 Evaluation</b></summary>

🔍 Reconstruction evaluation on validation data at `t=0`:

```bash
python sample.py --name darcy.repap.unet --gpu XXX
```

🎯 Reconstruction evaluation for a specific checkpoint:

```bash
python sample.py --name darcy.repap.unet --step 150000 --gpu XXX --num-batches 4 --save-images
```

🎲 Generative reverse-diffusion sampling from noise:

```bash
python sample.py --name darcy.repap.unet --step 150000 --gpu XXX --mode generative --num-samples 20 --use-ema --save-images
```

📁 Results are written to:

```text
trained_models/<run_name>/evaluation/step_<step>/
```

</details>

<details>
<summary><b>📦 Dependencies</b></summary>

Install dependencies with:

```bash
pip install -r requirements.txt
```

Main packages include `torch`, `numpy`, `pandas`, `matplotlib`, `tqdm`, `einops`, `torchvision`, `findiff`, `solidspy`, `scikit-image`, `pyyaml`, `imageio`, `opencv-python`, `scipy`, `scikit-learn`, `dill`, and optional `wandb`.

</details>
