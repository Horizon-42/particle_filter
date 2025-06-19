import numpy as np
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
