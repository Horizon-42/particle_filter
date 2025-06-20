import numpy as np
from math_utils import random_diagonal_cov

class NormalTransition:
    """
    Qt = At@Qt-1 + Bt@action_t-1 + Noise(Nomal Distributed)
    """
    delta_t = 1
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
    Q = random_diagonal_cov(4, 1000)


    @classmethod
    def propagate(cls, state:np.ndarray):
        return cls.A @ state + cls.B @ cls.action
    
    @classmethod
    def noisy_propagate(cls, state:np.ndarray):
        # wrong approach, reapte one noise
        # return cls.propagate(state) + np.random.multivariate_normal([0]*4, cls.Q).reshape(-1, 1)
        N_particles = state.shape[0]
        D_state = state.shape[1]

        noise = np.random.multivariate_normal(
            np.zeros(D_state), cls.Q, size=N_particles).reshape(N_particles, D_state, 1)
        return cls.propagate(state) + noise


if __name__ == "__main__":
    states = np.random.rand(10, 4, 1)
    transed = NormalTransition.propagate(states)
    print(transed)
    noise_transed = NormalTransition.noisy_propagate(states)
    print(np.mean(noise_transed**2-transed**2))
