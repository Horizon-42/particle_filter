import numpy as np
from math_utils import random_diagonal_cov


class BallTransition:
    """
    Qt = At@Qt-1 + Bt@action_t-1
    """

    def __init__(self, delta_t):
        self.A = np.array([[1, 0, delta_t, 0],
                           [0, 1, 0, delta_t],
                           [0, 0, 1, 0],
                           [0, 0, 0, 1]], dtype=float)
        self.B = np.array([
            [0, 0, 0],
            [0, 0.5*delta_t**2, 0],
            [0, 0, 0],
            [0, delta_t, 0]
        ], dtype=float)

        self.action = np.array([0, -10, 0]).reshape((3, 1))

    def propagate(self, state: np.ndarray):
        return self.A @ state + self.B @ self.action


class NormalTransition(BallTransition):
    """
    Qt = At@Qt-1 + Bt@action_t-1 + Noise(Nomal Distributed)
    """

    def __init__(self, delta_t):
        super().__init__(delta_t)
        # sigma for normal noise
        # Q = random_diagonal_cov(4, 1000)
        self.Q = np.array([
            [1000, 0, 0, 0],
            [0, 1000, 0, 0],
            [0, 0, 1000, 0],
            [0, 0, 0, 1000],
        ], dtype=float)

    def propagate(cls, state: np.ndarray):
        # wrong approach, reapte one noise
        # return cls.propagate(state) + np.random.multivariate_normal([0]*4, cls.Q).reshape(-1, 1)
        N_particles = state.shape[0]
        D_state = state.shape[1]
        Ball_num = state.shape[2]

        noises = np.zeros_like(state)
        for i in range(Ball_num):
            dn = np.random.multivariate_normal(
                np.zeros(D_state), cls.Q, size=N_particles)
            noises[:, :, i] = dn
        print(noises.shape)
        return super().propagate(state) + noises


if __name__ == "__main__":
    states = np.random.rand(10, 4, 2)*100
    transed = BallTransition(0.5).propagate(states)
    print(transed.shape)
    noise_transed = NormalTransition(0.5).propagate(states)
    print(np.mean(noise_transed**2-transed**2))
