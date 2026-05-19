"""
Poisson equation conditional generation training script

Task: Given charge density ρ, generate potential field U
Governing equation: (-Δ)U = ρ
Boundary condition: U|∂Ω = 0 (homogeneous Dirichlet)
"""

import os
import yaml
import argparse
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import colors
from matplotlib.lines import Line2D
import matplotlib.patheffects as path_effects
import torch
import torch.nn as nn
import torch.optim as optim
from pathlib import Path
from tqdm import tqdm
from torch.utils.data import DataLoader

from src.data_utils import cycle, generalized_image_to_b_xy_c, generalized_b_xy_c_to_image
from src.residuals_poisson_v2 import ResidualsPoissonV2

# ============== Paper-level style settings ==============
plt.rcParams.update({
    'text.usetex': False,
    'mathtext.fontset': 'cm',
    'font.family': 'serif',
    'font.size': 10,
    'axes.labelsize': 11,
    'axes.titlesize': 12,
    'legend.fontsize': 9,
    'figure.dpi': 150,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
    'axes.linewidth': 0.8,
})
from src.denoising_utils import (
    DenoisingDiffusion, EMA, save_model, load_model, 
    noop, exists, image_to_b_xy_c, b_xy_c_to_image,
    image_array_to_gif
)
from src.unet_model import Unet3D


# ============== Paper-level visualization functions ==============

def plot_potential_field(U, rho, title_suffix='', save_path=None):
    """Plot potential field (equipotential lines + electric field lines)"""
    fig, ax = plt.subplots(figsize=(6, 5))
    
    H, W = U.shape
    x = np.arange(W)
    y = np.arange(H)
    X, Y = np.meshgrid(x, y)
    
    # Equipotential surface fill
    levels = np.linspace(U.min(), U.max(), 25)
    cf = ax.contourf(X, Y, U, levels=levels, cmap='coolwarm', alpha=0.6)
    
    # Equipotential lines
    cn = ax.contour(X, Y, U, levels=levels[::2], colors='#2d2d2d', 
                    linewidths=0.5, alpha=0.7)
    
    # Electric field lines (E = -∇U)
    dx = 1.0
    grad_y, grad_x = np.gradient(U, dx, dx)
    Ex, Ey = -grad_x, -grad_y
    
    strm = ax.streamplot(X, Y, Ex, Ey, color='#1a5276', density=1.0,
                        linewidth=0.6, arrowsize=0.8)
    strm.lines.set_alpha(0.7)
    
    # Mark charge positions (extrema of ρ)
    rho_threshold = np.abs(rho).max() * 0.5
    pos_charges = np.where(rho > rho_threshold)
    neg_charges = np.where(rho < -rho_threshold)
    
    for cy, cx in zip(*pos_charges):
        ax.scatter(cx, cy, s=80, c='#c0392b', edgecolor='white', linewidth=1.2, zorder=10)
        ax.text(cx, cy, '+', fontsize=8, fontweight='bold', color='white',
                ha='center', va='center', zorder=11)
    
    for cy, cx in zip(*neg_charges):
        ax.scatter(cx, cy, s=80, c='#2980b9', edgecolor='white', linewidth=1.2, zorder=10)
        ax.text(cx, cy, '−', fontsize=8, fontweight='bold', color='white',
                ha='center', va='center', zorder=11)
    
    ax.set_xlabel(r'$x$')
    ax.set_ylabel(r'$y$')
    ax.set_aspect('equal')
    ax.set_xlim(0, W-1)
    ax.set_ylim(0, H-1)
    
    cbar = fig.colorbar(cf, ax=ax, shrink=0.85)
    cbar.set_label(r'Potential $U$')
    
    if title_suffix:
        ax.set_title(title_suffix)
    
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300, facecolor='white')
    plt.close(fig)
    return fig


def plot_comparison_paper(rho, U_gt, U_pred, l2_err, residual, save_path):
    """Paper-level comparison plot (4 subplots)"""
    fig, axes = plt.subplots(1, 4, figsize=(16, 4))
    
    H, W = U_gt.shape
    x = np.arange(W)
    y = np.arange(H)
    X, Y = np.meshgrid(x, y)
    
    # (a) Condition ρ
    ax = axes[0]
    max_abs_rho = np.abs(rho).max()
    norm_rho = colors.TwoSlopeNorm(vmin=-max_abs_rho, vcenter=0, vmax=max_abs_rho)
    im0 = ax.imshow(rho, cmap='RdBu_r', norm=norm_rho, origin='lower')
    ax.set_title(r'(a) Charge density $\rho$')
    ax.set_xlabel(r'$x$')
    ax.set_ylabel(r'$y$')
    cbar0 = fig.colorbar(im0, ax=ax, shrink=0.8)
    
    # (b) Ground truth U_gt
    ax = axes[1]
    levels = np.linspace(U_gt.min(), U_gt.max(), 20)
    cf1 = ax.contourf(X, Y, U_gt, levels=levels, cmap='coolwarm')
    ax.contour(X, Y, U_gt, levels=levels[::2], colors='k', linewidths=0.3, alpha=0.5)
    ax.set_title(r'(b) Ground truth $U$')
    ax.set_xlabel(r'$x$')
    ax.set_ylabel(r'$y$')
    ax.set_aspect('equal')
    cbar1 = fig.colorbar(cf1, ax=ax, shrink=0.8)
    
    # (c) Predicted U_pred
    ax = axes[2]
    cf2 = ax.contourf(X, Y, U_pred, levels=levels, cmap='coolwarm')
    ax.contour(X, Y, U_pred, levels=levels[::2], colors='k', linewidths=0.3, alpha=0.5)
    ax.set_title(f'(c) Predicted $U$\n$L_2$ error: {l2_err:.2e}')
    ax.set_xlabel(r'$x$')
    ax.set_ylabel(r'$y$')
    ax.set_aspect('equal')
    cbar2 = fig.colorbar(cf2, ax=ax, shrink=0.8)
    
    # (d) Error
    ax = axes[3]
    error = U_pred - U_gt
    max_abs_err = np.abs(error).max()
    norm_err = colors.TwoSlopeNorm(vmin=-max_abs_err, vcenter=0, vmax=max_abs_err)
    im3 = ax.imshow(error, cmap='RdBu_r', norm=norm_err, origin='lower')
    ax.set_title(f'(d) Error\nResidual: {residual:.2e}')
    ax.set_xlabel(r'$x$')
    ax.set_ylabel(r'$y$')
    cbar3 = fig.colorbar(im3, ax=ax, shrink=0.8)
    
    # Statistics
    stats_text = f'$\\|e\\|_\\infty$={np.abs(error).max():.2e}'
    axes[3].text(0.98, 0.02, stats_text, transform=axes[3].transAxes, fontsize=8,
                va='bottom', ha='right',
                bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.9))
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, facecolor='white')
    plt.close(fig)


def plot_training_curves(losses, output_dir, iteration):
    """Plot training curves"""
    # Check if projection head loss exists
    has_projection = 'projection' in losses and any(v > 0 for v in losses['projection'])
    n_cols = 4 if has_projection else 3
    fig, axes = plt.subplots(1, n_cols, figsize=(5*n_cols, 4))
    
    iters = list(range(len(losses['total'])))
    window = min(100, len(iters)//10 + 1)
    
    # Total loss
    axes[0].semilogy(iters, losses['total'], 'b-', linewidth=0.5, alpha=0.3)
    if len(iters) > window:
        smooth = np.convolve(losses['total'], np.ones(window)/window, mode='valid')
        axes[0].semilogy(range(window-1, len(iters)), smooth, 'b-', linewidth=1.5)
    axes[0].set_xlabel('Iteration')
    axes[0].set_ylabel('Total Loss')
    axes[0].set_title('Total Loss')
    axes[0].grid(True, alpha=0.3)
    
    # Data loss
    axes[1].semilogy(iters, losses['data'], 'g-', linewidth=0.5, alpha=0.3)
    if len(iters) > window:
        smooth = np.convolve(losses['data'], np.ones(window)/window, mode='valid')
        axes[1].semilogy(range(window-1, len(iters)), smooth, 'g-', linewidth=1.5)
    axes[1].set_xlabel('Iteration')
    axes[1].set_ylabel('Data Loss')
    axes[1].set_title('Data Loss (MSE)')
    axes[1].grid(True, alpha=0.3)
    
    # Residual
    axes[2].semilogy(iters, losses['residual'], 'r-', linewidth=0.5, alpha=0.3)
    if len(iters) > window:
        smooth = np.convolve(losses['residual'], np.ones(window)/window, mode='valid')
        axes[2].semilogy(range(window-1, len(iters)), smooth, 'r-', linewidth=1.5)
    axes[2].set_xlabel('Iteration')
    axes[2].set_ylabel('Residual')
    axes[2].set_title('PDE Residual')
    axes[2].grid(True, alpha=0.3)
    
    # Projection head loss (if exists)
    if has_projection:
        axes[3].semilogy(iters, losses['projection'], 'm-', linewidth=0.5, alpha=0.3)
        if len(iters) > window:
            smooth = np.convolve(losses['projection'], np.ones(window)/window, mode='valid')
            axes[3].semilogy(range(window-1, len(iters)), smooth, 'm-', linewidth=1.5)
        axes[3].set_xlabel('Iteration')
        axes[3].set_ylabel('Projection Loss')
        axes[3].set_title('Projection Head Residual')
        axes[3].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f'training_curves_{iteration}.png'), dpi=150)
    plt.close(fig)


# ============== Dataset class ==============

class PoissonDataset(torch.utils.data.Dataset):
    """Poisson dataset: load (ρ, U) pairs"""
    
    def __init__(self, data_dir, use_double=False):
        """
        Args:
            data_dir: Directory containing rho_data.csv and U_data.csv
            use_double: Whether to use double precision
        """
        rho_path = os.path.join(data_dir, 'rho_data.csv')
        U_path = os.path.join(data_dir, 'U_data.csv')
        
        print(f"Loading data from {data_dir}...")
        
        # Load data
        rho_data = pd.read_csv(rho_path, header=None).values
        U_data = pd.read_csv(U_path, header=None).values
        
        # Convert to tensor
        dtype = torch.float64 if use_double else torch.float32
        self.rho = torch.tensor(rho_data, dtype=dtype)
        self.U = torch.tensor(U_data, dtype=dtype)
        
        # Infer grid size
        n_pixels = self.rho.shape[1]
        self.pixels_per_dim = int(np.sqrt(n_pixels))
        assert self.pixels_per_dim ** 2 == n_pixels, "Data must be square grid"
        
        # Reshape to image format [N, C, H, W]
        N = self.rho.shape[0]
        self.rho = self.rho.reshape(N, 1, self.pixels_per_dim, self.pixels_per_dim)
        self.U = self.U.reshape(N, 1, self.pixels_per_dim, self.pixels_per_dim)
        
        # Concatenate: [N, 2, H, W] -> (rho, U)
        self.data = torch.cat([self.rho, self.U], dim=1)
        
        print(f"Loaded {N} samples, grid size: {self.pixels_per_dim}x{self.pixels_per_dim}")
        print(f"  rho range: [{self.rho.min():.4f}, {self.rho.max():.4f}]")
        print(f"  U range: [{self.U.min():.4f}, {self.U.max():.4f}]")
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        return self.data[idx]


# ============== Device setup ==============

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")


def parse_args():
    parser = argparse.ArgumentParser(description="Train Poisson conditional diffusion model")
    parser.add_argument("--config", type=str, default="model_poisson.yaml",
                        help="Path to config file")
    parser.add_argument("--fast", action="store_true",
                        help="Fast mode for quick experiments")
    parser.add_argument("--load_model", type=str, default=None,
                        help="Path to pretrained model")
    return parser.parse_args()


def main():
    args = parse_args()
    
    # Load configuration
    if os.path.exists(args.config):
        config = yaml.safe_load(Path(args.config).read_text())
    else:
        # Default configuration
        config = {
            'name': 'poisson_run_1',
            'diff_steps': 1000,
            'ddim_steps': 50,
            'x0_estimation': 'mean',  # 'mean' or 'sample'
            'c_data': 1.0,
            'c_residual': 0.01,
            'fd_acc': 2,
            'train_batch_size': 64,
            'train_iterations': 300000,
            'unet_dim': 64,
        }
    
    # Fast mode
    if args.fast:
        config['train_iterations'] = 50000
        config['train_batch_size'] = 32
        config['unet_dim'] = 32
        config['name'] = 'poisson_fast'
        print("Fast mode enabled!")
    
    name = config.get('name', 'poisson_run_1')
    wandb_track = config.get('wandb_track', False)
    
    # Diffusion parameters
    diff_steps = config.get('diff_steps', 1000)
    ddim_steps = config.get('ddim_steps', 50)
    use_ddim_x0 = config.get('x0_estimation', 'mean') == 'sample'
    
    # Loss weights
    c_data = config.get('c_data', 1.0)
    c_residual = config.get('c_residual', 0.01)
    
    # Projection head configuration
    use_projection_heads = config.get('use_projection_heads', False)
    projection_positions = config.get('projection_positions', ['encoder', 'bottleneck', 'decoder'])
    projection_hidden_dim = config.get('projection_hidden_dim', 0)
    c_projection = config.get('c_projection', 0.1)  # Projection head physical loss weight
    
    # Training parameters
    train_batch_size = config.get('train_batch_size', 64)
    train_iterations = config.get('train_iterations', 300000)
    unet_dim = config.get('unet_dim', 64)
    fd_acc = config.get('fd_acc', 2)
    
    # Evaluation parameters
    test_eval_freq = config.get('test_eval_freq', 500)
    sample_freq = config.get('sample_freq', 20000)
    save_freq = config.get('save_freq', 5000)  # Model save frequency
    plot_freq = config.get('plot_freq', 1000)  # Training curve plot frequency
    ema_start = config.get('ema_start', 1000)
    no_samples = config.get('no_samples', 8)
    save_output = True
    eval_residuals = True
    create_gif = config.get('create_gif', False)
    
    # Record training curves
    training_losses = {'total': [], 'data': [], 'residual': [], 'projection': []}
    
    # Grid parameters
    pixels_per_dim = 64
    domain_length = 1.0
    
    # Model parameters
    # Conditional generation: ρ (1ch) -> U (1ch)
    # Model input: U_t (1ch) + ρ (1ch) = 2ch
    # Model output: U (1ch)
    input_channels = 2  # U_t + ρ
    output_dim = 1  # U
    
    print(f"\n=== Configuration ===")
    print(f"Model: {name}")
    print(f"Diffusion steps: {diff_steps}")
    print(f"DDIM steps: {ddim_steps}")
    print(f"Use DDIM x0: {use_ddim_x0}")
    print(f"c_data: {c_data}, c_residual: {c_residual}")
    print(f"Batch size: {train_batch_size}")
    print(f"Train iterations: {train_iterations}")
    print(f"UNet dim: {unet_dim}")
    if use_projection_heads:
        print(f"Projection heads: {projection_positions}, hidden_dim: {projection_hidden_dim}, c_projection: {c_projection}")
    print()
    
    # ============== Data loading ==============
    
    data_dir_train = config.get('data_dir_train', './data/poisson_v2/train')
    data_dir_valid = config.get('data_dir_valid', './data/poisson_v2/valid')
    
    ds = PoissonDataset(data_dir_train)
    ds_valid = PoissonDataset(data_dir_valid)
    
    dl = cycle(DataLoader(ds, batch_size=train_batch_size, shuffle=True))
    dl_valid = cycle(DataLoader(ds_valid, batch_size=train_batch_size, shuffle=True))
    
    # ============== Model initialization ==============
    
    # UNet model
    model = Unet3D(
        dim=unet_dim,
        channels=input_channels,  # Input channels: U_t + ρ
        out_dim=output_dim,       # Output channels: U
        sigmoid_last_channel=False,
        # Projection head parameters (backward compatible)
        use_projection_heads=use_projection_heads,
        projection_positions=projection_positions,
        projection_hidden_dim=projection_hidden_dim,
    ).to(device)
    
    if args.load_model:
        load_model(args.load_model, model)
    
    # EMA
    ema = EMA(0.99)
    ema.register(model)
    
    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Number of trainable parameters: {num_params:,}")
    
    # Diffusion utilities
    diffusion_utils = DenoisingDiffusion(diff_steps, device, residual_grad_guidance=False)
    
    # Use normalized rho (v2 data) - always use V2 version
    use_normalized_rho = config.get('use_normalized_rho', True)  # Default to True for V2
    print(f"Using normalized rho (v2 data): rho stores charge q, will divide by h²")
    
    # Residual computer: always use ResidualsPoissonV2
    residuals = ResidualsPoissonV2(
        model=model,
        pixels_per_dim=pixels_per_dim,
        pixels_at_boundary=True,
        device=device,
        domain_length=domain_length,
        fd_acc=fd_acc,
        use_ddim_x0=use_ddim_x0,
        ddim_steps=ddim_steps,
    )
    
    # Optimizer
    optimizer = optim.Adam(model.parameters(), lr=1e-4)
    
    # Logging
    if wandb_track:
        import wandb
        wandb.init(project='pi_diffusion_poisson', name=name)
        log_fn = wandb.log
    else:
        log_fn = noop
    log_freq = 20
    
    # Output directory
    output_save_dir = f'./trained_models/{name}'
    os.makedirs(output_save_dir, exist_ok=True)
    
    # Save configuration
    with open(os.path.join(output_save_dir, 'config.yaml'), 'w') as f:
        yaml.dump(config, f)
    
    # ============== Training loop ==============
    
    print("\n=== Starting Training ===")
    pbar = tqdm(range(train_iterations + 1))
    
    for iteration in pbar:
        model.train()
        
        # Get batch data
        cur_batch = next(dl).to(device)  # [B, 2, H, W] = (rho, U)
        
        rho = cur_batch[:, 0:1]  # Condition: [B, 1, H, W]
        U_0 = cur_batch[:, 1:2]  # Target: [B, 1, H, W]
        
        batch_size = U_0.shape[0]
        
        # Random time step
        t = torch.randint(0, diff_steps, (batch_size,), device=device)
        
        # Diffusion forward process: U_t = sqrt(α̅_t) * U_0 + sqrt(1-α̅_t) * ε
        noise = torch.randn_like(U_0)
        a = diffusion_utils.diff_dict['alphas_bar_sqrt'][t].view(-1, 1, 1, 1)
        am1 = diffusion_utils.diff_dict['one_minus_alphas_bar_sqrt'][t].view(-1, 1, 1, 1)
        U_t = a * U_0 + am1 * noise
        
        # Condition concatenation: [B, 2, H, W] = (U_t, rho)
        model_input = torch.cat([U_t, rho], dim=1)
        
        # Convert to b_xy_c format
        model_input_flat = image_to_b_xy_c(model_input)
        
        # Compute residual and model output
        residual_input = ((model_input_flat, t), rho)
        out_dict = residuals.compute_residual(
            residual_input,
            reduce='per-batch',
            return_model_out=True,
            return_projections=use_projection_heads,
        )
        
        U_pred = out_dict['model_out']
        residual = out_dict['residual']
        projections = out_dict.get('projections', {})
        
        # Convert back to image format
        if len(U_pred.shape) == 3:
            U_pred = b_xy_c_to_image(U_pred)
        
        # Data loss (MSE)
        loss_data = nn.functional.mse_loss(U_pred, U_0)
        
        # Residual loss
        residual_loss_value = residual.abs().mean()
        
        # Projection head physical loss
        projection_loss_value = torch.tensor(0.0, device=device)
        if use_projection_heads and c_projection > 0. and projections:
            projection_losses = []
            for pos_name, proj_output in projections.items():
                # Convert projection output to format suitable for residual computation
                if len(proj_output.shape) == 4:  # Image format [B, C, H, W]
                    proj_input_reshaped = image_to_b_xy_c(proj_output)
                else:
                    continue
                
                # Compute physical residual for projection head
                try:
                    proj_residual_dict = residuals.compute_residual(
                        ((proj_input_reshaped, t), rho),
                        reduce='per-batch',
                        return_model_out=False,
                        skip_model_call=True,
                        given_model_output=proj_input_reshaped
                    )
                    proj_residual = proj_residual_dict['residual']
                    proj_loss = proj_residual.abs().mean()
                    projection_losses.append(proj_loss)
                except Exception as e:
                    pass  # Skip projection heads that fail to compute
            
            if projection_losses:
                projection_loss_value = torch.stack(projection_losses).mean()
        
        # Total loss
        loss = c_data * loss_data + c_residual * residual_loss_value
        if use_projection_heads and c_projection > 0.:
            loss = loss + c_projection * projection_loss_value
        
        # Backward propagation
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        
        # Record losses
        training_losses['total'].append(loss.item())
        training_losses['data'].append(loss_data.item())
        training_losses['residual'].append(residual_loss_value.item())
        training_losses['projection'].append(projection_loss_value.item() if isinstance(projection_loss_value, torch.Tensor) else projection_loss_value)
        
        # Logging
        if iteration % log_freq == 0:
            if use_projection_heads:
                pbar.set_description(
                    f"loss: {loss.item():.3e} | data: {loss_data.item():.3e} | res: {residual_loss_value.item():.3e} | proj: {projection_loss_value.item():.3e}"
                )
                log_fn({'projection_loss': projection_loss_value.item()}, step=iteration)
            else:
                pbar.set_description(
                    f"loss: {loss.item():.3e} | data: {loss_data.item():.3e} | res: {residual_loss_value.item():.3e}"
                )
            log_fn({'loss': loss.item()}, step=iteration)
            log_fn({'loss_data': loss_data.item()}, step=iteration)
            log_fn({'residual_mean_abs': residual_loss_value.item()}, step=iteration)
        
        # EMA update
        if iteration > ema_start:
            ema.update(model)
        
        # Periodic model checkpoint saving
        if iteration > 0 and iteration % save_freq == 0:
            save_model(config, model, iteration, output_save_dir)
            print(f"\n  [Checkpoint] Model saved at iteration {iteration}")
        
        # Periodic training curve plotting
        if iteration > 0 and iteration % plot_freq == 0:
            plot_training_curves(training_losses, output_save_dir, iteration)
        
        # ============== Validation ==============
        
        if iteration % test_eval_freq == 0 and iteration > 0:
            model.eval()
            ema.ema(residuals.model)
            
            with torch.no_grad():
                cur_test_batch = next(dl_valid).to(device)
                rho_test = cur_test_batch[:, 0:1]
                U_0_test = cur_test_batch[:, 1:2]
                
                t_test = torch.randint(0, diff_steps, (U_0_test.shape[0],), device=device)
                noise_test = torch.randn_like(U_0_test)
                a_test = diffusion_utils.diff_dict['alphas_bar_sqrt'][t_test].view(-1, 1, 1, 1)
                am1_test = diffusion_utils.diff_dict['one_minus_alphas_bar_sqrt'][t_test].view(-1, 1, 1, 1)
                U_t_test = a_test * U_0_test + am1_test * noise_test
                
                model_input_test = torch.cat([U_t_test, rho_test], dim=1)
                model_input_test_flat = image_to_b_xy_c(model_input_test)
                
                residual_input_test = ((model_input_test_flat, t_test), rho_test)
                out_dict_test = residuals.compute_residual(
                    residual_input_test,
                    reduce='per-batch',
                    return_model_out=True,
                )
                
                U_pred_test = out_dict_test['model_out']
                if len(U_pred_test.shape) == 3:
                    U_pred_test = b_xy_c_to_image(U_pred_test)
                
                loss_data_test = nn.functional.mse_loss(U_pred_test, U_0_test)
                residual_loss_test = out_dict_test['residual'].abs().mean()
                
                print(f"  [Valid] iteration {iteration}: data_loss={loss_data_test:.3e}, residual={residual_loss_test:.3e}")
                log_fn({'loss_data_test': loss_data_test.item()}, step=iteration)
                log_fn({'residual_mean_abs_test': residual_loss_test.item()}, step=iteration)
            
            ema.restore(residuals.model)
        
        # ============== Sampling evaluation ==============
        
        if (iteration % sample_freq == 0 and iteration > 0) or iteration == train_iterations:
            model.eval()
            ema.ema(residuals.model)
            
            # Get validation set conditions
            sample_batch = next(dl_valid).to(device)
            if sample_batch.shape[0] < no_samples:
                no_samples = sample_batch.shape[0]
            sample_batch = sample_batch[:no_samples]
            
            rho_cond = sample_batch[:, 0:1]  # Condition
            U_gt = sample_batch[:, 1:2]      # Ground truth
            
            # Conditional generation sampling
            print(f"\n  Generating {no_samples} samples...")
            conditioning_input = rho_cond
            sample_shape = (no_samples, output_dim, pixels_per_dim, pixels_per_dim)
            
            # Sampling loop
            cur_x = torch.randn(sample_shape, device=device)
            
            for i in tqdm(reversed(range(diff_steps)), desc="  Sampling", leave=False):
                # Condition concatenation
                model_input_sample = torch.cat([cur_x, rho_cond], dim=1)
                model_input_sample_flat = image_to_b_xy_c(model_input_sample)
                
                t_sample = torch.tensor([i], device=device)
                
                # Model prediction
                with torch.no_grad():
                    U_pred_sample = model(model_input_sample_flat, t_sample.repeat(no_samples))
                    if len(U_pred_sample.shape) == 3:
                        U_pred_sample = b_xy_c_to_image(U_pred_sample)
                
                # DDPM sampling step
                x0_pred = U_pred_sample
                
                # Compute mean
                posterior_mean_coef1 = diffusion_utils.diff_dict['posterior_mean_coef1'][i]
                posterior_mean_coef2 = diffusion_utils.diff_dict['posterior_mean_coef2'][i]
                mean = posterior_mean_coef1 * x0_pred + posterior_mean_coef2 * cur_x
                
                # Add noise (skip on last step)
                if i > 0:
                    noise = torch.randn_like(cur_x)
                    sigma = diffusion_utils.diff_dict['betas'][i].sqrt()
                    cur_x = mean + sigma * noise
                else:
                    cur_x = mean
            
            U_samples = cur_x  # Final sampling result
            
            # Compute sample residual
            residual_input_sample = ((image_to_b_xy_c(torch.cat([U_samples, rho_cond], dim=1)), 
                                      torch.zeros(no_samples, dtype=torch.long, device=device)), 
                                     rho_cond)
            sample_residual = residuals.compute_residual(
                U_samples, pass_through=True
            )['residual']
            sample_residual_mean = sample_residual.abs().mean(dim=1)
            
            # Compute L2 error
            l2_error = ((U_samples - U_gt) ** 2).mean(dim=(1, 2, 3)).sqrt()
            
            print(f"  Sample results:")
            print(f"    Mean L2 error: {l2_error.mean():.4e}")
            print(f"    Mean residual: {sample_residual_mean.mean():.4e}")
            
            log_fn({'sample_l2_error': l2_error.mean().item()}, step=iteration)
            log_fn({'sample_residual': sample_residual_mean.mean().item()}, step=iteration)
            
            # Save visualizations
            output_save_dir_step = os.path.join(output_save_dir, f'step_{iteration}')
            os.makedirs(output_save_dir_step, exist_ok=True)
            
            for idx in range(min(4, no_samples)):  # Save first 4 samples
                rho_np = rho_cond[idx, 0].cpu().numpy()
                U_gt_np = U_gt[idx, 0].cpu().numpy()
                U_pred_np = U_samples[idx, 0].cpu().numpy()
                
                # Paper-level comparison plot
                plot_comparison_paper(
                    rho_np, U_gt_np, U_pred_np,
                    l2_error[idx].item(), sample_residual_mean[idx].item(),
                    os.path.join(output_save_dir_step, f'comparison_{idx}.png')
                )
                
                # Potential field visualization (with electric field lines)
                plot_potential_field(
                    U_pred_np, rho_np,
                    title_suffix=f'Generated (iter {iteration})',
                    save_path=os.path.join(output_save_dir_step, f'potential_{idx}.png')
                )
                
                # Ground truth potential field
                plot_potential_field(
                    U_gt_np, rho_np,
                    title_suffix='Ground Truth',
                    save_path=os.path.join(output_save_dir_step, f'potential_gt_{idx}.png')
                )
            
            # Save statistics
            stats_df = pd.DataFrame({
                'sample_idx': list(range(no_samples)),
                'l2_error': l2_error.cpu().numpy(),
                'residual': sample_residual_mean.cpu().numpy(),
            })
            stats_df.to_csv(os.path.join(output_save_dir_step, 'statistics.csv'), index=False)
            
            # Save model
            if iteration > 0:
                save_model(config, model, iteration, output_save_dir)
            
            ema.restore(residuals.model)
    
    print("\n=== Training Complete ===")
    
    if wandb_track:
        import wandb
        wandb.finish()


if __name__ == "__main__":
    main()
