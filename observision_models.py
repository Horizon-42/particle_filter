import numpy as np
from math_utils import random_diagonal_cov
from scipy.stats import multivariate_normal


class NormalObservation:
    C = np.array([[1, 0, 0, 0],
                  [0, 1, 0, 0]], dtype=float)

    R = random_diagonal_cov(2, 100)

    noise_distribution = multivariate_normal([0]*2, R)

    @classmethod
    def observe(cls, state: np.ndarray):
        observation = cls.C @ state
        return observation + cls.noise_distribution.rvs().reshape(-1, 1)

    @classmethod
    def evaluation(cls, observe: np.ndarray, state: np.ndarray):
        expected_observe = cls.C@state
        return multivariate_normal.pdf(
            x=observe.flatten(), mean=expected_observe.flatten(), cov=cls.R)


if __name__ == "__main__":
    state = np.random.rand(10, 4, 1)*100
    # print(state)
    observe = NormalObservation.observe(state)
    print(observe.shape)

    probs = [NormalObservation.evaluation(
        observe[i], state[i, :]) for i in range(observe.shape[0])]
    print(len(probs))
