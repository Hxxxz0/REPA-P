"""
Poisson equation data generation script v2
Generate (charge density ρ, potential U) data pairs

Improvement: Store normalized ρ (charge q), not q/h²
This makes both ρ and U have numerical ranges around [-2, 2]

Governing equation: (-Δ)U = ρ
Boundary condition: U|∂Ω = 0 (homogeneous Dirichlet)
Grid: 64×64 (including boundary), h = 1/63
Solver: Discrete Sine Transform (DST)
"""

import os
import argparse
import numpy as np
from tqdm import tqdm


# ============== DST Solver ==============

def dst1(x):
    """Type-I Discrete Sine Transform"""
    x = np.asarray(x)
    n = x.shape[-1]
    y = np.zeros(x.shape[:-1] + (2 * (n + 1),), float)
    y[..., 1:n+1] = x
    y[..., n+2:] = -x[..., ::-1]
    Y = np.fft.fft(y, axis=-1)
    return -Y.imag[..., 1:n+1]


def idst1(X):
    """Type-I 离散正弦逆变换"""
    return dst1(X) / (2 * (X.shape[-1] + 1))


def solve_poisson_dst(rho, h):
    """
    使用 DST 求解 Poisson 方程: -ΔU = ρ
    
    Args:
        rho: 内部节点上的源项, shape (N, N)
        h: 网格间距
    
    Returns:
        U: 内部节点上的解, shape (N, N)
    """
    N = rho.shape[0]
    rho_hat = dst1(dst1(rho.T).T)
    m = np.arange(1, N + 1)
    n = np.arange(1, N + 1)
    lam_m = 2.0 * (1 - np.cos(np.pi * m / (N + 1))) / (h * h)
    lam_n = 2.0 * (1 - np.cos(np.pi * n / (N + 1))) / (h * h)
    lam_2d = lam_m[:, None] + lam_n[None, :]
    U_hat = rho_hat / lam_2d
    return idst1(idst1(U_hat.T).T)


def random_charge(rng, q_min=0.5, q_max=1.5):
    """生成随机电荷量 (带随机正负号)"""
    sign = rng.choice([-1.0, 1.0])
    mag = rng.uniform(q_min, q_max)
    return sign * mag


# ============== 数据生成 ==============

def generate_single_sample(rng, N_full=64, K=2, q_min=0.5, q_max=1.5):
    """
    生成单个样本 (ρ, U)
    
    关键改进:
    - rho_normalized: 存储电荷量 q（不是 q/h²）
    - U: 电势场（与之前相同）
    
    这样 rho 和 U 的数值范围都在合理区间
    
    Args:
        rng: numpy random generator
        N_full: 含边界的网格大小 (64)
        K: 点电荷数量 (2)
        q_min, q_max: 电荷量范围
    
    Returns:
        rho_normalized: 归一化电荷密度场 (存储 q 值), shape (N_full, N_full)
        U_full: 电势场, shape (N_full, N_full)
    """
    N_inner = N_full - 2  # 内部节点数 = 62
    h = 1.0 / (N_full - 1)  # 网格间距 = 1/63
    
    # 在内部节点上放置点电荷
    # 用于求解的 rho (需要 q/h²)
    rho_inner_solver = np.zeros((N_inner, N_inner), dtype=np.float64)
    # 用于存储的 rho (只存储 q)
    rho_inner_normalized = np.zeros((N_inner, N_inner), dtype=np.float64)
    
    for _ in range(K):
        q = random_charge(rng, q_min, q_max)
        # 随机选择内部节点位置
        i = rng.integers(0, N_inner)
        j = rng.integers(0, N_inner)
        # 求解器需要: q / h^2
        rho_inner_solver[i, j] += q / (h * h)
        # 存储归一化值: 直接存储 q
        rho_inner_normalized[i, j] += q
    
    # 求解 Poisson 方程
    U_inner = solve_poisson_dst(rho_inner_solver, h)
    
    # 扩展到含边界的 64x64 网格 (边界值为 0)
    rho_full = np.zeros((N_full, N_full), dtype=np.float64)
    rho_full[1:-1, 1:-1] = rho_inner_normalized  # 存储归一化的 rho
    
    U_full = np.zeros((N_full, N_full), dtype=np.float64)
    U_full[1:-1, 1:-1] = U_inner
    
    return rho_full, U_full


def generate_dataset(num_samples, N_full=64, K=2, q_min=0.5, q_max=1.5, seed=None):
    """
    批量生成数据集
    """
    rng = np.random.default_rng(seed)
    
    rho_list = []
    U_list = []
    
    for _ in tqdm(range(num_samples), desc="Generating samples"):
        rho, U = generate_single_sample(rng, N_full, K, q_min, q_max)
        rho_list.append(rho.flatten())
        U_list.append(U.flatten())
    
    rho_data = np.stack(rho_list, axis=0)
    U_data = np.stack(U_list, axis=0)
    
    return rho_data, U_data


def save_dataset(rho_data, U_data, output_dir):
    """保存数据集为 CSV 格式"""
    os.makedirs(output_dir, exist_ok=True)
    
    rho_path = os.path.join(output_dir, 'rho_data.csv')
    U_path = os.path.join(output_dir, 'U_data.csv')
    
    print(f"Saving rho_data to {rho_path}...")
    np.savetxt(rho_path, rho_data, delimiter=',', fmt='%.10e')
    
    print(f"Saving U_data to {U_path}...")
    np.savetxt(U_path, U_data, delimiter=',', fmt='%.10e')
    
    # 打印统计信息
    print(f"\n数据统计:")
    print(f"  rho shape: {rho_data.shape}")
    print(f"  rho range: [{rho_data.min():.4f}, {rho_data.max():.4f}]")
    print(f"  U shape: {U_data.shape}")
    print(f"  U range: [{U_data.min():.4f}, {U_data.max():.4f}]")


def main():
    parser = argparse.ArgumentParser(description="Generate Poisson equation dataset (v2 - normalized)")
    parser.add_argument("--train_samples", type=int, default=50000, 
                        help="Number of training samples")
    parser.add_argument("--valid_samples", type=int, default=2048, 
                        help="Number of validation samples")
    parser.add_argument("--N", type=int, default=64, 
                        help="Grid size (including boundary)")
    parser.add_argument("--K", type=int, default=2, 
                        help="Number of point charges per sample")
    parser.add_argument("--q_min", type=float, default=0.5, 
                        help="Minimum charge magnitude")
    parser.add_argument("--q_max", type=float, default=1.5, 
                        help="Maximum charge magnitude")
    parser.add_argument("--output_dir", type=str, default="./data/poisson_v2", 
                        help="Output directory")
    parser.add_argument("--seed", type=int, default=42, 
                        help="Random seed")
    args = parser.parse_args()
    
    print(f"="*60)
    print("Poisson 数据生成 v2 (归一化版本)")
    print(f"="*60)
    print(f"\n配置:")
    print(f"  网格大小: {args.N}x{args.N}")
    print(f"  点电荷数量: {args.K}")
    print(f"  电荷量范围: [{args.q_min}, {args.q_max}]")
    print(f"  训练样本: {args.train_samples}")
    print(f"  验证样本: {args.valid_samples}")
    print(f"  输出目录: {args.output_dir}")
    print()
    
    # 生成训练集
    print("=== 生成训练集 ===")
    rho_train, U_train = generate_dataset(
        args.train_samples, args.N, args.K, args.q_min, args.q_max, 
        seed=args.seed
    )
    save_dataset(rho_train, U_train, os.path.join(args.output_dir, 'train'))
    
    # 生成验证集
    print("\n=== 生成验证集 ===")
    rho_valid, U_valid = generate_dataset(
        args.valid_samples, args.N, args.K, args.q_min, args.q_max, 
        seed=args.seed + 1000
    )
    save_dataset(rho_valid, U_valid, os.path.join(args.output_dir, 'valid'))
    
    print("\n=== 数据生成完成! ===")
    print(f"\n关键改进:")
    print(f"  - rho 现在存储电荷量 q，不是 q/h²")
    print(f"  - rho 范围 ~[-3, 3]，与 U 范围 ~[-2, 2] 匹配")
    print(f"  - 训练时需要使用新的残差计算 (乘以 h²)")


if __name__ == "__main__":
    main()
