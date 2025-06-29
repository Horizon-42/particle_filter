import numpy as np
import matplotlib.pyplot as plt

# ------------------------------
# Step 1: 定义目标分布（混合高斯）
# ------------------------------

def target_distribution(x):
    """
    目标分布是两个二维高斯分布的混合。
    参数：
        x: np.ndarray, shape=(2,), 要计算的二维点
    返回：
        概率密度值（未归一化）
    """
    # 第一个高斯分布：均值(-2, -2)，协方差单位阵
    mu1 = np.array([-2, -2])
    cov1 = np.eye(2)

    # 第二个高斯分布：均值(2, 2)，协方差单位阵
    mu2 = np.array([2, 2])
    cov2 = np.eye(2)

    # 高斯密度公式（不需要归一化常数）
    def gaussian(x, mu, cov):
        diff = x - mu
        return np.exp(-0.5 * diff.T @ np.linalg.inv(cov) @ diff)

    # 混合权重 0.5 : 0.5
    return 0.5 * gaussian(x, mu1, cov1) + 0.5 * gaussian(x, mu2, cov2)


# ------------------------------
# Step 2: Metropolis-Hastings MCMC 主过程
# ------------------------------

def metropolis_hastings(target_fn, initial_state, num_samples, proposal_std=1.0):
    """
    执行 Metropolis-Hastings MCMC 采样。
    
    参数：
        target_fn: 目标分布函数（返回非标准化密度）
        initial_state: np.ndarray, 初始采样点
        num_samples: int, 要生成的样本数量
        proposal_std: float, 提议分布的标准差（用于随机步长）
    
    返回：
        samples: np.ndarray, shape=(num_samples, dim), MCMC采样结果
    """
    dim = initial_state.shape[0]
    samples = np.zeros((num_samples, dim))  # 存储所有采样点
    current_state = initial_state.copy()    # 当前粒子位置
    current_prob = target_fn(current_state) # 当前粒子的目标分布密度

    for i in range(num_samples):
        # Step 1: 从提议分布中采样新点（对称提议分布，正态分布）
        proposal = current_state + np.random.normal(0, proposal_std, size=dim)

        # Step 2: 计算新点的目标密度
        proposal_prob = target_fn(proposal)

        # Step 3: 计算接受率 alpha
        acceptance_ratio = proposal_prob / current_prob
        alpha = min(1.0, acceptance_ratio)

        # Step 4: 随机接受或拒绝
        if np.random.rand() < alpha:
            current_state = proposal
            current_prob = proposal_prob

        # Step 5: 存储当前（可能更新后的）粒子
        samples[i] = current_state

    return samples


# ------------------------------
# Step 3: 运行采样并可视化
# ------------------------------

if __name__ == "__main__":
    np.random.seed(42)  # 设置随机种子以便复现
    initial = np.array([0.0, 0.0])
    samples = metropolis_hastings(target_distribution, initial, num_samples=10000, proposal_std=1.0)

    # 可视化采样结果
    plt.figure(figsize=(8, 6))
    plt.scatter(samples[:, 0], samples[:, 1], alpha=0.2, s=5, color='blue')
    plt.title("Metropolis-Hastings MCMC Sampling of a Gaussian Mixture")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.axis('equal')
    plt.grid(True)
    plt.show()
