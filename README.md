<h1 align="center">REPA-P for Physics-Informed Diffusion Models</h1>

<h4 align="center">U-Net code for PIDM baseline and REPA-P</h4>

$~$

## Introduction & Setup

This repository contains the U-Net implementation used in our REPA-P experiments, built on top of Physics-Informed Diffusion Models (PIDM).

The code supports both methods with the same training script:

```yaml
use_projection_heads: False  # PIDM baseline
use_projection_heads: True   # REPA-P
```

We provide two main scripts:

`main.py` trains the model. Change the dataset, run name, and projection-head switch in `model.yaml`, then run:

```bash
python main.py --config model.yaml --gpu 0
```

`sample.py` evaluates trained models. It reads the saved `model.yaml` from the checkpoint folder, so the same script works for both PIDM and REPA-P:

```bash
python sample.py --name darcy.repap.unet --gpu 0
```

## Data

Datasets and checkpoints are not included. Download the data separately and place it as follows:

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
    └── ...
```

The `charge` task is generated synthetically and does not require downloaded data.

## How to Run

All settings are in `model.yaml`.

Choose one dataset:

```yaml
gov_eqs: darcy       # darcy, mechanics, charge, turbulent
```

Choose PIDM baseline:

```yaml
run_name: darcy.pidm.unet
use_projection_heads: False
projection_positions: []
projection_hidden_dim: 0
c_projection: 0.0
```

Or choose REPA-P:

```yaml
run_name: darcy.repap.unet
use_projection_heads: True
projection_positions: decoder
projection_hidden_dim: 128
c_projection: 0.01
```

Then train:

```bash
python main.py --config model.yaml --gpu 0
```

Training outputs are saved to:

```text
trained_models/<run_name>/
```

## Dataset Settings

Use these typical settings in `model.yaml`:

```yaml
# Darcy
gov_eqs: darcy
train_iterations: 150000
train_batch_size: 64
c_ineq: 0.0
lambda_opt: 0.0
```

```yaml
# Mechanics
gov_eqs: mechanics
train_iterations: 300000
train_batch_size: 8
c_ineq: 0.001
lambda_opt: 0.1
```

```yaml
# Charge
gov_eqs: charge
train_iterations: 150000
train_batch_size: 64
c_ineq: 0.0
lambda_opt: 0.0
```

```yaml
# Turbulent
gov_eqs: turbulent
train_iterations: 150000
train_batch_size: 64
turbulent_data_path: ./data/ch_2Dxysec.pickle
```

## Evaluation

Evaluate the latest checkpoint:

```bash
python sample.py --name darcy.repap.unet --gpu 0
```

Evaluate a specific checkpoint:

```bash
python sample.py --name darcy.repap.unet --step 150000 --gpu 0 --num-batches 4 --save-images
```

Results are written to:

```text
trained_models/<run_name>/evaluation/step_<step>/
```

## Dependencies

Install dependencies with:

```bash
pip install -r requirements.txt
```

Main packages include `torch`, `numpy`, `pandas`, `matplotlib`, `tqdm`, `einops`, `torchvision`, `findiff`, `solidspy`, `scikit-image`, `pyyaml`, and `imageio`.

## Notes

This release is U-Net only. DiT code, datasets, checkpoints, logs, and generated images are intentionally excluded. Do not upload `data/` or `trained_models/` to arXiv.

## Citation

If this code is useful for your research, please cite PIDM and our accompanying paper.
