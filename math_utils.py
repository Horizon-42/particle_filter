import numpy as np
from numpy.linalg import inv, det

def random_cov(dim, scale=100):
    L = np.tril(np.random.rand(dim, dim))*scale
    cov = L@L.T
    return cov


def build_cov(var: list[float]):
    dim = len(var)
    cov = np.eye(dim)
    np.fill_diagonal(cov, var)
    return cov


def random_diagonal_cov(dim: int, scale: float = 100):
    var = np.random.rand(dim)*scale
    cov = np.eye(dim)
    np.fill_diagonal(cov, var)
    return cov


def multivariate_normal_pdf_vectorized(x: np.ndarray, means: np.ndarray, cov: np.ndarray):
    d = x.shape[0]
    cov_inv = inv(cov)
    cov_det = det(cov)
    norm_const = 1.0 / np.sqrt((2 * np.pi) ** d * cov_det)

    diffs = means - x  # shape: (N, D)
    exponents = -0.5 * np.sum(diffs @ cov_inv *
                              diffs, axis=1)  # shape: (N,)
    return norm_const * np.exp(exponents)


def multivariate_normal_logpdf_vectorized(x: np.ndarray, means: np.ndarray, cov: np.ndarray):
    """
    计算多元正态分布的对数概率密度函数 (log PDF)。
    此函数针对一个观测点 x 和多个均值 (means) 进行向量化计算。

    参数:
        x (np.ndarray): 单个观测向量，形状为 (D,)，D 是维度。
        means (np.ndarray): 多个均值向量，形状为 (N, D)，N 是粒子数量。
        cov (np.ndarray): 协方差矩阵，形状为 (D, D)。

    返回:
        np.ndarray: 对数概率密度数组，形状为 (N,)。
    """
    d = x.shape[0]  # 维度

    # 确保协方差矩阵是正定的，并且计算其逆和行列式
    cov_inv = inv(cov)
    cov_det = det(cov)

    # 检查行列式，如果非常接近0或负数，可能存在数值问题
    if cov_det <= 0:
        # 可以选择返回 -np.inf 或者抛出错误
        # print("Warning: Covariance matrix determinant is non-positive, returning -inf for log-PDF.")
        return np.full(means.shape[0], -np.inf)  # 返回所有粒子的log-PDF为-inf

    # 对数化归一化常数
    # log(1 / sqrt((2 * pi)^d * cov_det))
    # = -0.5 * log((2 * pi)^d * cov_det)
    # = -0.5 * (d * log(2 * pi) + log(cov_det))
    log_norm_const = -0.5 * (d * np.log(2 * np.pi) + np.log(cov_det))

    # 计算差值 (mean_i - x)
    # x 的形状是 (D,)，means 的形状是 (N, D)
    # NumPy 会自动广播 x 到 (N, D) 来执行减法
    diffs = means - x  # 结果形状: (N, D)

    # 计算指数项
    # (diffs @ cov_inv) 的形状是 (N, D)
    # (diffs @ cov_inv * diffs) 是元素级乘法，形状仍是 (N, D)
    # np.sum(..., axis=1) 对每个粒子求和，结果形状是 (N,)
    exponents = -0.5 * np.sum(diffs @ cov_inv * diffs, axis=1)  # 结果形状: (N,)

    # 对数概率密度是 对数归一化常数 + 指数项
    # log(P(x)) = log(Norm_const * exp(Exponent_term)) = log(Norm_const) + Exponent_term
    log_pdf_values = log_norm_const + exponents

    return log_pdf_values


def sample_points_in_circle(center, radius, n_points):
    """
    Uniformly sample 2D points inside a circle.

    Parameters:
        center: (x, y) tuple of the circle center
        radius: radius of the circle
        n_points: number of points to generate

    Returns:
        points: (n_points, 2) array of sampled points
    """
    theta = np.random.uniform(0, 2*np.pi, n_points)
    r = radius * np.sqrt(np.random.uniform(0, 1, n_points))  # √ for uniformity

    x = center[0] + r * np.cos(theta)
    y = center[1] + r * np.sin(theta)

    return np.stack((x, y), axis=-1)
