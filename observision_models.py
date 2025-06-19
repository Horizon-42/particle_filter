import numpy as np
from utils import random_diagonal_cov
from scipy.stats import multivariate_normal

class NormalObservation:
    C = np.array([[1, 0, 0, 0],
                  [0, 1, 0, 0]], dtype=float)
    
    R = random_diagonal_cov(2, 100)

    noise_distribution = multivariate_normal([0]*2, R)

    @classmethod
    def observe(cls, state:np.ndarray):
        observation = cls.C @ state
        return observation + cls.noise_distribution.rvs().reshape(-1, 1)
    
    @classmethod
    def evaluation(cls, observe:np.ndarray, state:np.ndarray):
        return multivariate_normal.pdf(x=observe.flatten(), mean=state.flatten(), cov=cls.R)
