import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
from scipy.stats import norm
from copy import deepcopy
from scipy.stats.distributions import chi2
import numpy as np

# 绘制协方差2D椭圆
def plot_covariance_ellipse(ax, mean, cov, confidence_level=1, color='black', alpha=0.2, label=None):
    """
    在给定轴上绘制一个2D协方差椭圆。
    参数:
        ax: Matplotlib 的 axes 对象。
        mean: 2D 均值向量 (例如, [x, y])。
        cov: 2x2 协方差矩阵。
        confidence_level: 所需的置信水平 (例如, 0.95 代表 95% 置信)。
        color: 椭圆的颜色。
        alpha: 椭圆的透明度。
        label: 椭圆的标签。
    """
    # 确保协方差矩阵是2x2
    if cov.shape != (2, 2):
        raise ValueError("Covariance matrix must be 2x2 for 2D ellipse plotting.")

    # 计算置信水平对应的卡方分布的临界值
    # 对于 2D 高斯分布，马氏距离平方服从自由度为 2 的卡方分布
    # s 是椭圆半轴的缩放因子
    s = np.sqrt(chi2.ppf(confidence_level, 2))

    # 对协方差矩阵进行特征值分解
    eigenvalues, eigenvectors = np.linalg.eig(cov)

    # 计算椭圆的旋转角度 (以度为单位)
    # arctan2 给出的是弧度，转换为度
    angle = np.degrees(np.arctan2(eigenvectors[1, 0], eigenvectors[0, 0]))

    # 计算椭圆的宽度和高度 (半轴长度乘以缩放因子 s)
    width, height = 2 * s * np.sqrt(eigenvalues)

    # 创建椭圆补丁对象
    ellipse = Ellipse(xy=mean, width=width, height=height, angle=angle,
                      facecolor=color, edgecolor=color, alpha=alpha, label=label)
    ax.add_patch(ellipse)
    return ellipse

def ploat_observations(ax, q, o):
    """
    在给定的轴上绘制状态和观测点。
    参数:
        ax: Matplotlib 的 axes 对象。
        q: 状态数组 (N x 4)，每行表示一个状态 [x, y, vx, vy]。
        o: 观测点数组 (M x 2)，每行表示一个观测点 [x, y]。
    """
    x = q[:, 0]
    y = q[:, 1]
    ax.plot(x, y, '-', color='g', label='State Trajectory')
    
    ox = o[:, 0]
    oy = o[:, 1]
    ax.scatter(ox, oy, s=15, color='r', label='Observations')

    ax.set_xlabel('X Position')
    ax.set_ylabel('Y Position')
    ax.set_title('State and Observations')
    ax.legend()
    ax.grid(True)
    ax.set_aspect('equal', adjustable='box') # 保持坐标轴比例一致

def plot_throwing(q, o = None, q_preds = None, sigma_preds = None):
    x = q[:,0]
    y = q[:,1]
    vx = q[:,2]
    vy = q[:,3]

    fig, ax = plt.subplots(figsize=(12, 9))

    ax.plot(x, y, '-', color='g')
    # plt.quiver(x, y, vx, np.zeros_like(vx), angles='xy', scale_units='xy', scale=2, color='g', label='Velocity vectors')
    # plt.quiver(x, y, np.zeros_like(vy), vy, angles='xy', scale_units='xy', scale=2, color='y', label='Velocity vectors')
    # plt.legend()
    ax.set_xlabel('X Position')
    ax.set_ylabel('Y Position')
    ax.set_title('Kalman Filter Trajectory with 95% Confidence Ellipses')
    # ax.legend()
    ax.grid(True)
    ax.set_aspect('equal', adjustable='box') # 保持坐标轴比例一致，避免椭圆变形

    if o is not None:
        ox = o[:, 0]
        oy = o[:, 1]
        ax.scatter(ox, oy, s=15, color='r')
    
    if q_preds is not None:
        xp = q_preds[:,0]
        yp = q_preds[:,1]
        ax.plot(xp, yp, '*--', color='b')

        if sigma_preds is not None:
            for i in range(len(xp)):
                pos_mean = [xp[i], yp[i]]
                pos_cov = sigma_preds[i, :2,:2]

                plot_covariance_ellipse(ax, pos_mean, pos_cov, confidence_level=0.95,
                                    color='green', alpha=0.15) # 使用绿色，半透明

    plt.show()

def plot_particles(ax, particles, weights, color='blue'):
    """
    在给定的轴上绘制粒子。
    参数:
        ax: Matplotlib 的 axes 对象。
        particles: 粒子状态数组 (N x 4)，每行表示一个粒子 [x, y, vx, vy]。
        color: 粒子的颜色。
        alpha: 粒子的透明度。
    """
    x, y = particles[:, 0], particles[:, 1]
    vx, vy = particles[:, 2], particles[:, 3]
    max_weight = np.max(weights)
    alpha = weights / max_weight  # 归一化权重到 [0, 1] 范围
    alpha = np.clip(alpha, 0, 1)  # 确保 alpha 在 [0, 1] 范围内
    ax.scatter(x, y, color='red', alpha=alpha, s=1)
    ax.quiver(x, y, vx, vy, angles='xy', scale_units='xy', scale=3, color=color, alpha=alpha)