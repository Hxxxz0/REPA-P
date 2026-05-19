# Learning to Think in Physics: Breaking Shortcut Learning in Scientific Diffusion via Representation Alignment

Physics-informed diffusion model for solving PDEs. Demonstrates representation alignment by enforcing physical constraints on intermediate latent features through projection heads, improving generalization and physical consistency.

## Quick Start

```bash
# Install dependencies
pip install -r requirements.txt

# Generate training data
python src/data_generation_poisson_v2.py

# Train baseline model
python main_poisson.py --config model_poisson_v2_bs8.yaml

# Train model with projection heads (ours)
python main_poisson.py --config model_poisson_v2_bs8_proj_bottleneck.yaml

# Evaluate
python sample_poisson.py \
    --checkpoint trained_models/poisson_v2_bs8_proj_bottleneck/model/checkpoint_20000.pt \
    --config trained_models/poisson_v2_bs8_proj_bottleneck/config.yaml \
    --output_dir results/bottleneck
```

## Dependencies

```bash
pip install -r requirements.txt
```

Required packages:
- `torch`
- `numpy`
- `pandas`
- `matplotlib`
- `tqdm`
- `pyyaml`
- `einops`

## Usage

This repository contains a toy experiment demonstrating the concept on **Electrostatic Charge Potential generation** governed by the Poisson equation: $(- \Delta)U = \rho$.

### 1. Data Generation

First, generate the training and validation data (pairs of charge density $\rho$ and potential $U$). This uses the V2 data generation script which properly normalizes physical quantities.

```bash
# Generate data (default: 20000 training samples, 2000 validation samples)
python src/data_generation_poisson_v2.py
```

Data will be saved to `data/poisson_v2/`.

### 2. Training

You can train two versions of the model for comparison:

**A. Baseline (Standard Diffusion Model)**
This model uses standard diffusion training without internal physical constraints.

```bash
python main_poisson.py --config model_poisson_v2_bs8.yaml
```

**B. Ours (Physical Representation Alignment)**
This model adds a projection head at the **bottleneck** layer, forcing the latent features to be decodable into physically valid states (low PDE residual).

```bash
python main_poisson.py --config model_poisson_v2_bs8_proj_bottleneck.yaml
```

**Training Logs & Checkpoints:**
- Models are saved to `trained_models/`
- Training curves are saved as PNGs in the same directory
- Checkpoints are saved every 5000 iterations (configurable via `save_freq`)

### 3. Sampling & Evaluation

To evaluate a trained model and visualize the results (including physical consistency):

```bash
# Evaluate Baseline
python sample_poisson.py \
    --checkpoint trained_models/poisson_v2_bs8/model/checkpoint_20000.pt \
    --config trained_models/poisson_v2_bs8/config.yaml \
    --output_dir results/baseline

# Evaluate Ours (Bottleneck)
python sample_poisson.py \
    --checkpoint trained_models/poisson_v2_bs8_proj_bottleneck/model/checkpoint_20000.pt \
    --config trained_models/poisson_v2_bs8_proj_bottleneck/config.yaml \
    --output_dir results/bottleneck
```

This will generate:
- Comparison plots (Ground Truth vs Generated)
- Visualization of Potential Field ($U$) and Charge Density ($\rho$)
- Statistics (L2 Error, PDE Residual)

