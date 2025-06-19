import numpy as np
from math_utils import random_diagonal_cov

class NormalTransition:
    """
    Qt = At@Qt-1 + Bt@action_t-1 + Noise(Nomal Distributed)
    """
    delta_t = 0.5
    A = np.array([[1, 0, delta_t, 0],
                  [0, 1, 0, delta_t],
                  [0, 0, 1, 0],
                  [0, 0, 0, 1]], dtype=float)

    B = np.array([
        [0, 0, 0],
        [0, 0.5*delta_t**2, 0],
        [0, 0, 0],
        [0, delta_t, 0]
    ], dtype=float)

    # assume g == 10 for convinent
    action = np.array([0, -10, 0]).reshape((3,1))

    # sigma for normal noise
    Q = random_diagonal_cov(4, 50)

    @classmethod
    def propagate(cls, state:np.ndarray):
        return cls.A @ state + cls.B @ cls.action
    
    @classmethod
    def noisy_propagate(cls, state:np.ndarray):
        return cls.propagate(state) + np.random.multivariate_normal([0]*4, cls.Q).reshape(-1, 1)


if __name__ == "__main__":
    states = np.random.rand(10, 4, 1)
    transed = NormalTransition.propagate(states)
    print(transed)
