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
