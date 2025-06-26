import numpy as np
from math_utils import random_diagonal_cov, sample_points_in_circle
from scipy.stats import t
from enum import Enum


class TransitionType(Enum):
    Normal = 1
    Uniform = 2
    StudentT = 3

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

    def propagate(self, states: np.ndarray):
        return self.A @ states + self.B @ self.action


class NormalTransition(BallTransition):
    """
    Qt = At@Qt-1 + Bt@action_t-1 + Noise(Nomal Distributed)
    """

    def __init__(self, delta_t):
        super().__init__(delta_t)
        # sigma for normal noise
        self.Q = random_diagonal_cov(4, 2000)

    def propagate(self, states: np.ndarray):
        N_particles = states.shape[0]
        D_state = states.shape[1]
        N_ball = states.shape[2]

        noises = np.random.multivariate_normal(
            np.zeros(D_state), self.Q, size=N_particles*N_ball).reshape(N_particles, D_state, N_ball)
        return super().propagate(states) + noises


class UniformTransition(BallTransition):
    def __init__(self, delta_t):
        super().__init__(delta_t)

        # position R
        self.position_R = 60

        # speed R
        self.speed_R = 50

    def propagate(self, states: np.ndarray):
        N_particles = states.shape[0]
        D_state = states.shape[1]
        N_ball = states.shape[2]

        pos_noise = sample_points_in_circle(
            (0, 0), self.position_R, N_particles*N_ball).reshape(N_particles, 2, N_ball)
        # pos_noise = np.zeros(shape=(N_particles, 2, N_ball))
        speed_noise = sample_points_in_circle(
            (0, 0), self.speed_R, N_particles*N_ball).reshape(N_particles, 2, N_ball)
        return super().propagate(states) + np.concatenate([pos_noise, speed_noise], axis=1)


class StudentTTransition(BallTransition):
    def __init__(self, delta_t, v: float = 2, scale: float = 10):
        super().__init__(delta_t)

        self.v = v
        self.scale = scale

    def propagate(self, states: np.ndarray):
        N_particles = states.shape[0]
        D_state = states.shape[1]
        N_ball = states.shape[2]

        return super().propagate(states) + t.rvs(df=self.v, scale=self.scale, size=N_particles*D_state*N_ball).reshape(states.shape)


if __name__ == "__main__":
    states = np.random.rand(5, 4, 3)*100
    real_trans = BallTransition(0.5)
    transed_states = real_trans.propagate(states)
    # print(transed_states)
    # print(transed_states.shape)

    uniform_noise_trans = UniformTransition(0.5)
    noised_states = uniform_noise_trans.propagate(states)
