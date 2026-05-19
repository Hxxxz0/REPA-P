"""
Poisson equation conditional generation sampling script
Load trained model, generate U given rho
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
import torch
from pathlib import Path
from tqdm import tqdm
from torch.utils.data import DataLoader

from src.denoising_utils import (
    DenoisingDiffusion, load_model, 
    image_to_b_xy_c, b_xy_c_to_image
)
from src.unet_model import Unet3D
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

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")


class PoissonDataset(torch.utils.data.Dataset):
    """Poisson dataset"""
    def __init__(self, data_dir):
        rho_path = os.path.join(data_dir, 'rho_data.csv')
        U_path = os.path.join(data_dir, 'U_data.csv')
        
        rho_data = pd.read_csv(rho_path, header=None).values
        U_data = pd.read_csv(U_path, header=None).values
        
        self.rho = torch.tensor(rho_data, dtype=torch.float32)
        self.U = torch.tensor(U_data, dtype=torch.float32)
        
        n_pixels = self.rho.shape[1]
        self.pixels_per_dim = int(np.sqrt(n_pixels))
        
        N = self.rho.shape[0]
        self.rho = self.rho.reshape(N, 1, self.pixels_per_dim, self.pixels_per_dim)
        self.U = self.U.reshape(N, 1, self.pixels_per_dim, self.pixels_per_dim)
        self.data = torch.cat([self.rho, self.U], dim=1)
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        return self.data[idx]


def plot_comparison(rho, U_gt, U_pred, l2_err, residual, save_path):
    """Comparison plot - pure black and white minimalist style, white background"""
    fig, axes = plt.subplots(1, 4, figsize=(16, 4), facecolor='white')
    for ax in axes:
        ax.set_facecolor('white')
    
    H, W = U_gt.shape
    x = np.arange(W)
    y = np.arange(H)
    X, Y = np.meshgrid(x, y)
    
    # Unified equipotential line levels
    vmin_U = min(U_gt.min(), U_pred.min())
    vmax_U = max(U_gt.max(), U_pred.max())
    levels = np.linspace(vmin_U, vmax_U, 12)
    
    # Charge positions
    rho_threshold = np.abs(rho).max() * 0.3
    pos_charges = np.where(rho > rho_threshold) if rho_threshold > 0 else ([], [])
    neg_charges = np.where(rho < -rho_threshold) if rho_threshold > 0 else ([], [])
    
    def add_charges_bw(ax):
        # Positive charges: white background with black +
        for cy, cx in zip(*pos_charges):
            ax.scatter(cx, cy, s=60, c='white', edgecolor='black', linewidth=1.2, zorder=10)
            ax.text(cx, cy, '+', fontsize=7, fontweight='bold', color='black', ha='center', va='center', zorder=11)
        # Negative charges: black background with white −
        for cy, cx in zip(*neg_charges):
            ax.scatter(cx, cy, s=60, c='black', edgecolor='black', linewidth=1.2, zorder=10)
            ax.text(cx, cy, '−', fontsize=7, fontweight='bold', color='white', ha='center', va='center', zorder=11)
    
    def setup_ax(ax):
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_xlim(0, W-1)
        ax.set_ylim(0, H-1)
        ax.set_aspect('equal')
        for spine in ax.spines.values():
            spine.set_linewidth(0.8)
            spine.set_color('black')
    
    # (a) rho - charge density (grayscale)
    ax = axes[0]
    ax.imshow(rho, cmap='gray', origin='lower')
    add_charges_bw(ax)
    setup_ax(ax)
    ax.set_title(r'(a) $\rho$', fontsize=10, pad=6)
    
    # (b) U_gt - Ground Truth
    ax = axes[1]
    ax.contour(X, Y, U_gt, levels=levels, colors='black', linewidths=0.5)
    grad_y, grad_x = np.gradient(U_gt, 1.0, 1.0)
    ax.streamplot(X, Y, -grad_x, -grad_y, color='#555555', density=0.6, linewidth=0.35, arrowsize=0.5)
    add_charges_bw(ax)
    setup_ax(ax)
    ax.set_title(r'(b) $U$ (GT)', fontsize=10, pad=6)
    
    # (c) U_pred - generated result
    ax = axes[2]
    ax.contour(X, Y, U_pred, levels=levels, colors='black', linewidths=0.5)
    grad_y, grad_x = np.gradient(U_pred, 1.0, 1.0)
    ax.streamplot(X, Y, -grad_x, -grad_y, color='#555555', density=0.6, linewidth=0.35, arrowsize=0.5)
    add_charges_bw(ax)
    setup_ax(ax)
    ax.set_title(r'(c) $U$ (Gen)', fontsize=10, pad=6)
    
    # (d) Error - error (grayscale)
    ax = axes[3]
    error = U_pred - U_gt
    ax.imshow(error, cmap='gray', origin='lower')
    ax.contour(X, Y, error, levels=[0], colors='black', linewidths=1, linestyles='--')
    setup_ax(ax)
    ax.set_title(r'(d) Error', fontsize=10, pad=6)
    
    # Bottom statistics
    fig.text(0.5, 0.02, f'$L_2$ error: {l2_err:.2e}  |  Residual: {residual:.2e}', 
             ha='center', fontsize=10, color='black')
    
    plt.tight_layout(rect=[0, 0.06, 1, 1])
    plt.savefig(save_path, dpi=300, facecolor='white', bbox_inches='tight')
    plt.close(fig)


def plot_potential_field(U, rho, title_suffix='', save_path=None):
    """Potential field visualization - clean unified style"""
    fig, ax = plt.subplots(figsize=(4, 4))
    
    # Normalize to [0, 1]
    U_normalized = (U - U.min()) / (U.max() - U.min() + 1e-8)
    image = np.uint8(U_normalized * 255)
    
    ax.imshow(image, cmap='gray', vmin=0, vmax=255, origin='lower')
    ax.axis('off')
    
    if title_suffix:
        ax.set_title(title_suffix, fontsize=11, pad=5, color='green')
    
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300, facecolor='white', bbox_inches='tight', pad_inches=0)
    plt.close(fig)


def save_single_field(field, title='', save_path=None, rho=None, show_physics=True):
    """Save single field visualization - pure black and white minimalist style, white background"""
    fig, ax = plt.subplots(figsize=(4.5, 4.5), facecolor='white')
    ax.set_facecolor('white')
    
    H, W = field.shape
    x = np.arange(W)
    y = np.arange(H)
    X, Y = np.meshgrid(x, y)
    
    if show_physics and rho is not None:
        # === Equipotential lines (black) ===
        levels = np.linspace(field.min(), field.max(), 15)
        ax.contour(X, Y, field, levels=levels, colors='black', linewidths=0.6)
        
        # === Electric field lines (gray) ===
        dx = 1.0
        grad_y, grad_x = np.gradient(field, dx, dx)
        Ex, Ey = -grad_x, -grad_y
        strm = ax.streamplot(X, Y, Ex, Ey, color='#555555', density=0.8,
                            linewidth=0.4, arrowsize=0.6)
        
        # === Charge markers (black and white) ===
        rho_threshold = np.abs(rho).max() * 0.3
        if rho_threshold > 0:
            pos_charges = np.where(rho > rho_threshold)
            neg_charges = np.where(rho < -rho_threshold)
            
            # Positive charges: white background with black edge and black +
            for cy, cx in zip(*pos_charges):
                ax.scatter(cx, cy, s=80, c='white', edgecolor='black', 
                          linewidth=1.5, zorder=10, marker='o')
                ax.text(cx, cy, '+', fontsize=9, fontweight='bold', color='black',
                       ha='center', va='center', zorder=11)
            
            # Negative charges: black background with white −
            for cy, cx in zip(*neg_charges):
                ax.scatter(cx, cy, s=80, c='black', edgecolor='black',
                          linewidth=1.5, zorder=10, marker='o')
                ax.text(cx, cy, '−', fontsize=9, fontweight='bold', color='white',
                       ha='center', va='center', zorder=11)
        
        # Minimalist style: remove ticks but keep border
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_xlim(0, W-1)
        ax.set_ylim(0, H-1)
        ax.set_aspect('equal')
        
        # Thin black border
        for spine in ax.spines.values():
            spine.set_linewidth(0.8)
            spine.set_color('black')
    else:
        # Simple grayscale image
        ax.imshow(field, cmap='gray', origin='lower')
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_aspect('equal')
        for spine in ax.spines.values():
            spine.set_linewidth(0.8)
            spine.set_color('black')
    
    if title:
        ax.set_title(title, fontsize=10, color='black', pad=8)
    
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300, facecolor='white', edgecolor='none', bbox_inches='tight', pad_inches=0.02)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description="Sample from trained Poisson model")
    parser.add_argument("--checkpoint", type=str, 
                        default="./trained_models/poisson_bs8/model/checkpoint_10000.pt",
                        help="Path to checkpoint")
    parser.add_argument("--config", type=str,
                        default="./trained_models/poisson_bs8/config.yaml",
                        help="Path to config")
    parser.add_argument("--data_dir", type=str,
                        default="./data/poisson_v2/valid",
                        help="Data directory for conditions")
    parser.add_argument("--num_samples", type=int, default=8,
                        help="Number of samples to generate")
    parser.add_argument("--output_dir", type=str, default=None,
                        help="Output directory")
    args = parser.parse_args()
    
    # 加载配置
    if os.path.exists(args.config):
        config = yaml.safe_load(Path(args.config).read_text())
    else:
        config = {}
    
    # 参数
    diff_steps = config.get('diff_steps', 1000)
    unet_dim = config.get('unet_dim', 64)
    pixels_per_dim = 64
    domain_length = 1.0
    fd_acc = config.get('fd_acc', 2)
    
    print(f"\n=== Configuration ===")
    print(f"Checkpoint: {args.checkpoint}")
    print(f"Diffusion steps: {diff_steps}")
    print(f"UNet dim: {unet_dim}")
    
    # 输出目录
    if args.output_dir is None:
        checkpoint_dir = os.path.dirname(os.path.dirname(args.checkpoint))
        step = args.checkpoint.split('_')[-1].replace('.pt', '')
        args.output_dir = os.path.join(checkpoint_dir, f'samples_step_{step}')
    os.makedirs(args.output_dir, exist_ok=True)
    print(f"Output directory: {args.output_dir}")
    
    # Load data
    print(f"\nLoading data from {args.data_dir}...")
    ds = PoissonDataset(args.data_dir)
    dl = DataLoader(ds, batch_size=args.num_samples, shuffle=True)
    
    # Initialize model
    input_channels = 2  # U_t + rho
    output_dim = 1  # U
    
    model = Unet3D(
        dim=unet_dim,
        channels=input_channels,
        out_dim=output_dim,
        sigmoid_last_channel=False,
    ).to(device)
    
    # Load weights
    load_model(args.checkpoint, model)
    model.eval()
    
    num_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {num_params:,}")
    
    # Diffusion utilities
    diffusion_utils = DenoisingDiffusion(diff_steps, device, residual_grad_guidance=False)
    
    # Residual computer
    residuals = ResidualsPoissonV2(
        model=model,
        pixels_per_dim=pixels_per_dim,
        pixels_at_boundary=True,
        device=device,
        domain_length=domain_length,
        fd_acc=fd_acc,
        use_ddim_x0=False,
        ddim_steps=0,
    )
    
    # Get condition data
    sample_batch = next(iter(dl)).to(device)
    no_samples = min(args.num_samples, sample_batch.shape[0])
    sample_batch = sample_batch[:no_samples]
    
    rho_cond = sample_batch[:, 0:1]  # [B, 1, H, W]
    U_gt = sample_batch[:, 1:2]      # [B, 1, H, W]
    
    print(f"\n=== Generating {no_samples} samples ===")
    
    # Sampling
    sample_shape = (no_samples, output_dim, pixels_per_dim, pixels_per_dim)
    cur_x = torch.randn(sample_shape, device=device)
    
    with torch.no_grad():
        for i in tqdm(reversed(range(diff_steps)), desc="Sampling"):
            # Condition concatenation
            model_input = torch.cat([cur_x, rho_cond], dim=1)
            model_input_flat = image_to_b_xy_c(model_input)
            
            t = torch.tensor([i], device=device).repeat(no_samples)
            
            # Model prediction x0
            U_pred = model(model_input_flat, t)
            if len(U_pred.shape) == 3:
                U_pred = b_xy_c_to_image(U_pred)
            
            x0_pred = U_pred
            
            # DDPM sampling
            posterior_mean_coef1 = diffusion_utils.diff_dict['posterior_mean_coef1'][i]
            posterior_mean_coef2 = diffusion_utils.diff_dict['posterior_mean_coef2'][i]
            mean = posterior_mean_coef1 * x0_pred + posterior_mean_coef2 * cur_x
            
            if i > 0:
                noise = torch.randn_like(cur_x)
                sigma = diffusion_utils.diff_dict['betas'][i].sqrt()
                cur_x = mean + sigma * noise
            else:
                cur_x = mean
    
    U_samples = cur_x
    
    # Compute residual
    sample_residual = residuals.compute_residual(U_samples, pass_through=True)['residual']
    sample_residual_mean = sample_residual.abs().mean(dim=1)
    
    # Compute L2 error
    l2_error = ((U_samples - U_gt) ** 2).mean(dim=(1, 2, 3)).sqrt()
    
    print(f"\n=== Results ===")
    print(f"Mean L2 error: {l2_error.mean():.4e}")
    print(f"Mean residual: {sample_residual_mean.mean():.4e}")
    
    # Save visualizations
    print(f"\nSaving visualizations to {args.output_dir}...")
    
    for idx in range(no_samples):
        rho_np = rho_cond[idx, 0].cpu().numpy()
        U_gt_np = U_gt[idx, 0].cpu().numpy()
        U_pred_np = U_samples[idx, 0].cpu().numpy()
        
        # Comparison plot
        plot_comparison(
            rho_np, U_gt_np, U_pred_np,
            l2_error[idx].item(), sample_residual_mean[idx].item(),
            os.path.join(args.output_dir, f'comparison_{idx}.png')
        )
        
        # Save each channel separately - preserve physical features in minimalist style
        # Generated U (with equipotential lines, electric field lines, charge markers)
        save_single_field(
            U_pred_np,
            title=f'Generated (res: {sample_residual_mean[idx].item():.2e})',
            save_path=os.path.join(args.output_dir, f'sample_sample_{idx}_0.png'),
            rho=rho_np,
            show_physics=True
        )
        
        # Ground truth U (with equipotential lines, electric field lines, charge markers)
        save_single_field(
            U_gt_np,
            title='Ground Truth',
            save_path=os.path.join(args.output_dir, f'sample_gt_{idx}_0.png'),
            rho=rho_np,
            show_physics=True
        )
        
        # rho (condition input) - simple display
        save_single_field(
            rho_np,
            title=r'Charge density $\rho$',
            save_path=os.path.join(args.output_dir, f'sample_cond_{idx}_0.png'),
            rho=None,
            show_physics=False
        )
        
        # Save CSV data (consistent with other models)
        sample_dir = os.path.join(args.output_dir, f'sample_{idx}')
        os.makedirs(sample_dir, exist_ok=True)
        np.savetxt(os.path.join(sample_dir, 'sample_0.csv'), U_pred_np, delimiter=',')
        np.savetxt(os.path.join(sample_dir, 'gt_0.csv'), U_gt_np, delimiter=',')
        np.savetxt(os.path.join(sample_dir, 'cond_0.csv'), rho_np, delimiter=',')
    
    # Save statistics
    stats_df = pd.DataFrame({
        'sample_idx': list(range(no_samples)),
        'l2_error': l2_error.cpu().numpy(),
        'residual': sample_residual_mean.cpu().numpy(),
    })
    stats_df.to_csv(os.path.join(args.output_dir, 'statistics.csv'), index=False)
    
    print(f"\n=== Done! ===")
    print(f"Statistics saved to {os.path.join(args.output_dir, 'statistics.csv')}")


if __name__ == "__main__":
    main()
