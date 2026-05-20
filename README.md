<div align="center">

# Learning to Think in Physics: Breaking Shortcut Learning in Scientific Diffusion via Representation Alignment

**Accepted by ICML 2026**

[![License](https://img.shields.io/badge/license-MIT-blue.svg)](./LICENSE)
[![Python](https://img.shields.io/badge/python-3.10+-blue.svg)](https://docs.python.org/3/)
[![PyTorch](https://img.shields.io/badge/pytorch-2.0+-ee4c2c.svg)](https://pytorch.org/)

<img src="assets/repa_p_overview.png" width="100%" alt="REPA-P Overview"/>

</div>

## 🗞️ News
- **`2026-05-01`**: 🔥🔥🔥 Code of **REPA-P** released! Welcome to Star! ⭐
- **`2026-01-23`**: 🔥🔥 **REPA-P** accepted to **ICML 2026**!

## 📌 Overview

**REPA-P** is a teacher-free, architecture-agnostic framework that aligns intermediate representations with physical states in diffusion models. It attaches lightweight **1×1 projection heads** to selected layers, decodes hidden activations into physical quantities, and applies PDE residual losses during training. These heads are **discarded at inference**, introducing **zero overhead**.

REPA-P compels the network to *"think in physics"* rather than memorizing statistical shortcuts, leading to faster convergence, lower physics residuals, and stronger out-of-distribution generalization across multiple PDE benchmarks.

## 🎯 Highlights
- ⚡ **Teacher-free & lightweight** — only needs 1×1 conv heads, no external teacher model
- 🧠 **Breaks shortcut learning** — forces intermediate features to encode valid physical states
- 🚀 **Zero inference overhead** — projection heads discarded after training
- 🔧 **Architecture-agnostic** — consistent gains on both U-Net and Diffusion Transformer (DiT)
- 🌊 **Four PDE benchmarks** — Darcy flow, topology optimization (mechanics), Poisson equation, and turbulent channel flow

## ⚙️ Installation

```bash
git clone https://github.com/Hxxxz0/REPA-P.git
cd REPA-P
pip install -r requirements.txt
```

## 🚀 Quick Start

```bash
# Train
python main.py --config configs/model.darcy.pidm.yaml --gpu 0

# Evaluate (reconstruction)
python sample.py --name darcy.pidm.unet --step 150000 --gpu 0 --mode reconstruction

# Evaluate (generative)
python sample.py --name darcy.repap.unet --step 150000 --gpu 0 --mode generative --num-samples 20 --use-ema --save-images
```

## 📂 Benchmarks & Configs

| Benchmark | PIDM Config | REPA-P Config | Type |
|-----------|-------------|---------------|------|
| 🌊 Darcy Flow | `model.darcy.pidm.yaml` | `model.darcy.repap.yaml` | Unconditional |
| ⚡ Poisson | `model.poisson_pidm.yaml` | `model.poisson_repap.yaml` | Conditional |
| 🏗️ Mechanics | `model.mechanics.yaml` | *(set `use_projection_heads: True`)* | Conditional |
| 🌀 Turbulent | `model.turbulent.yaml` | *(set `use_projection_heads: True`)* | Unconditional |

Toggle REPA-P by setting in your config:

```yaml
use_projection_heads: True
projection_positions: [decoder]   # encoder / bottleneck / decoder / output
projection_hidden_dim: 128
c_projection: 0.01
```

For detailed parameter descriptions, see the reference config [`model.yaml`](model.yaml).

## 📊 Data

### 🌊 Darcy Flow
Download from [ETHZ Research Collection](https://doi.org/10.3929/ethz-b-000674074). Place `p_data.csv` and `K_data.csv` under `data/darcy/train/` and `data/darcy/valid/`.

### ⚡ Poisson Equation
```bash
python src_pr/data_generation_poisson.py --train_samples 2000 --valid_samples 200 --output_dir ./data/poisson
```

### 🏗️ Mechanics
Download from [ETHZ Research Collection](https://doi.org/10.3929/ethz-b-000674074). Place under `data/mechanics/`.

### 🌀 Turbulent Flow
Place the processed channel-flow pickle at `data/ch_2Dxysec.pickle`.

## 🛠️ Usage

**Training** — pick a config and run:

```bash
python main.py --config configs/model.darcy.pidm.yaml --gpu 0
```

Checkpoints and logs are saved to `trained_models/<run_name>/`.

**Evaluation** — reconstruction (t=0 denoising) or generative sampling from noise:

```bash
python sample.py --name <run_name> --step <checkpoint_step> --gpu 0 --mode reconstruction
python sample.py --name <run_name> --step <checkpoint_step> --gpu 0 --mode generative --use-ema --save-images
```

## 📄 Citation

```bibtex
@inproceedings{jia2026learning,
  title={Learning to Think in Physics: Breaking Shortcut Learning in
         Scientific Diffusion via Representation Alignment},
  author={Jia, Haozhe and Yin, Pengyu and Chen, Wenshuo and Liang, Shaofeng
          and Wang, Lei and Tian, Bowen and Wang, Xiucheng and Jia, Nanqian
          and Yue, Yutao},
  booktitle={International Conference on Machine Learning (ICML)},
  year={2026}
}
```

## 📜 License

This project is licensed under the [MIT License](./LICENSE).
