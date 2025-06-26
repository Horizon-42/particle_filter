import numpy as np
from math_utils import random_diagonal_cov, multivariate_normal_pdf_vectorized, multivariate_normal_logpdf_vectorized
from scipy.stats import multivariate_normal
from utils import plot_observations
from scipy.stats import t
from itertools import permutations
from scipy.optimize import linear_sum_assignment
from scipy.special import logsumexp
from scipy.spatial.distance import cdist


# Remind: The observation model shouldn't know which observ come from which ball

class BallObservation:
    def __init__(self, ball_num: int):
        self.C = np.array([[1, 0, 0, 0],
                  [0, 1, 0, 0]], dtype=float)
        self.ball_num = ball_num

    def observe(self, state: np.ndarray):
        return self.C@state

class NormalObservation(BallObservation):
    def __init__(self, ball_num: int, var_scale: float = 10):
        super().__init__(ball_num)

        # self.R = random_diagonal_cov(self.d_observe, var_scale)
        self.R = np.eye(2)
        np.fill_diagonal(self.R, var_scale)

        self.noise_distribution = multivariate_normal(
            np.zeros(2), self.R)

    def observe(self, state: np.ndarray):
        particle_num = state.shape[0]
        noise = self.noise_distribution.rvs(size=particle_num*self.ball_num)
        return super().observe(state) + noise.reshape(particle_num, 2, self.ball_num)

    def log_likelihoods(self, single_observe: np.ndarray, states: np.ndarray):
        """
        Evaluates the likelihood of an observation given multiple particle states.

        Args:
            observe (np.ndarray): The current 2D observation (e.g., [z_x, z_y]).
                                  Should be a 1D or 2D array (e.g., (2,) or (1, 2)).
            states (np.ndarray): A 3D array of particle states (e.g., (N, 4, 1) if states are column vectors,
                                 or (N, 4) if states are row vectors).

        Returns:
            np.array: An array of normlized weights

        """
        # This assumes states are (N, D_state, 1) or (N, D_state)
        # If states are (N, D_state, 1) -> (N, D_obs, 1)
        # print(f"observe dimension:{self.d_observe}")
        expected_observations = super().observe(states)
        # print(expected_observations)
        # print(f"Expected observation shape:{expected_observations.shape}")
        expected_observations = expected_observations.reshape(
            -1, 2)
        # print(expected_observations)
        # print(f"Expected observation shape:{expected_observations.shape}")
        # print(f"Single observation shape:{single_observe.shape}")
        # print(f"R shape:{self.R.shape}")

        # print(f"single observ:{single_observe}")
        # print(
        #     f"single observ reshaped:{single_observe.reshape(-1, self.d_observe, order='F').flatten()}")

        log_likelihoods = multivariate_normal_logpdf_vectorized(
            single_observe.flatten(), expected_observations, self.R)
        # print(f"log_likelihoods.shape: {log_likelihoods.shape}")
        # print(
        #     f"loglikelihood max:{np.max(log_likelihoods)}, min{np.min(log_likelihoods)}, mean:{np.mean(log_likelihoods)}")
        # print(log_likelihoods)
        return log_likelihoods

    def evaluation(self, single_observe: np.ndarray, states: np.ndarray):
        log_likelihoods = np.zeros(shape=(states.shape[0], 1))
        for i in range(self.ball_num):
            # joint probality is multiply, but here we use log, so add
            log_likelihoods[:, 0] += self.log_likelihoods(
                single_observe[:, i], states[:, :, i:i+1])

        max_log_likelihood = np.max(log_likelihoods)

        # 将对数似然转换为非归一化权重，通过减去最大值避免exp(大正数)溢出
        # np.exp(log(w_i) - log(w_max)) = w_i / w_max
        unnormalized_weights = np.exp(log_likelihoods - max_log_likelihood)
        # print(f"exp unnormalized_weights:{unnormalized_weights}")

        return unnormalized_weights / np.sum(unnormalized_weights)


def sinkhorn_log_assignment(cost_matrix, epsilon=0.01, max_iter=100, tol=1e-3):
    """
    Sinkhorn-Knopp algorithm to compute soft assignment.
    cost_matrix: shape (n, m) -- distance matrix
    Returns: soft assignment matrix of shape (n, m)
    """
    n, m = cost_matrix.shape
    K = np.exp(-cost_matrix / epsilon)  # Gibbs kernel
    u = np.ones(n)
    v = np.ones(m)

    for _ in range(max_iter):
        u_prev = u.copy()
        u = 1.0 / (K @ v)
        v = 1.0 / (K.T @ u)
        if np.linalg.norm(u - u_prev, 1) < tol:
            break

    # Final transport matrix (soft-assignment)
    P = np.diag(u) @ K @ np.diag(v)
    return P
class StudentTObservation(BallObservation):
    """
    Student's t-distribution
    """

    def __init__(self, ball_num: int, v: float = 0.5, scale: float = 10):
        super().__init__(ball_num)
        # freedom, less, fatter
        self.v = v
        self.scale = scale

    def observe(self, state: np.ndarray):
        particle_num = state.shape[0]
        # assume x, y is independent, and every ball is independent
        return super().observe(state) + t.rvs(df=self.v, scale=self.scale, size=particle_num*2*self.ball_num).reshape(particle_num, 2, self.ball_num)

    def evaluation0(self, single_observe: np.ndarray, states: np.ndarray):
        N = states.shape[0]
        expected_observations = super().observe(states).reshape(N, -1, order='F')

        observs = single_observe.reshape(-1, order='F')
        log_likelihoods = t.logpdf(
            observs, df=self.v, scale=self.scale, loc=expected_observations)
        log_likelihoods = np.mean(log_likelihoods, axis=1)

        max_log_likelihood = np.max(log_likelihoods)

        # 将对数似然转换为非归一化权重，通过减去最大值避免exp(大正数)溢出
        # np.exp(log(w_i) - log(w_max)) = w_i / w_max
        unnormalized_weights = np.exp(log_likelihoods - max_log_likelihood)
        # print(f"exp unnormalized_weights:{unnormalized_weights}")

        return unnormalized_weights / np.sum(unnormalized_weights)

    def evaluation(self, single_observe: np.ndarray, states: np.ndarray):
        N = states.shape[0]  # 粒子数
        B = self.ball_num    # 预测小球数
        O = single_observe.shape[-1]  # 观测小球数（允许 O ≠ B）

        # 粒子预测：形状 (N, 2, B)
        predicted_positions = super().observe(states)

        # 观测点 (2, O) → (O, 2)
        observed_points = single_observe.T  # (O, 2)

        log_likelihoods = np.zeros(N)

        for obs in observed_points:
            # obs 是形状 (2,) 的观测点

            # 对每个预测点 i，计算 log P(obs | p_i) ，预测点形状: (N, 2, B)
            pred_x = predicted_positions[:, 0, :]  # (N, B)
            pred_y = predicted_positions[:, 1, :]  # (N, B)

            logpdf_x = t.logpdf(obs[0], df=self.v,
                                scale=self.scale, loc=pred_x)  # (N, B)
            logpdf_y = t.logpdf(obs[1], df=self.v,
                                scale=self.scale, loc=pred_y)  # (N, B)

            logpdf = logpdf_x + logpdf_y  # (N, B)

            # soft assignment: log-sum-exp over predicted balls
            # 使用 logsumexp for numerical stability
            log_prob = logsumexp(logpdf - np.log(B), axis=1)  # (N,)
            log_likelihoods += log_prob

        # Normalize
        max_log_likelihood = np.max(log_likelihoods)
        weights = np.exp(log_likelihoods - max_log_likelihood)
        return weights / np.sum(weights)

    def evaluation2(self, single_observe: np.ndarray, states: np.ndarray):
        N_particles = states.shape[0]
        # shape: (N, 2, ball_num_predicted)
        predicted_positions = super().observe(states)

        ball_num_observed = single_observe.shape[-1]
        observe_points = single_observe.T  # shape: (ball_num_observed, 2)

        weights = np.zeros(N_particles)

        for i in range(N_particles):
            # shape: (ball_num_predicted, 2)
            pred_points = predicted_positions[i].T

            # Compute distance matrix (observed x predicted)
            cost_matrix = cdist(observe_points, pred_points,
                                metric='sqeuclidean')

            # Get soft assignment matrix (observed x predicted)
            soft_assignment = sinkhorn_log_assignment(
                cost_matrix, epsilon=0.05)

            # Each (o_i, p_j) has a soft match weight soft_assignment[i, j]
            # Now compute weighted likelihood
            log_likelihood = 0.0
            for obs_i in range(ball_num_observed):
                for pred_j in range(pred_points.shape[0]):
                    loc = pred_points[pred_j]
                    obs = observe_points[obs_i]
                    log_prob = t.logpdf(obs[0], df=self.v, scale=self.scale, loc=loc[0]) + \
                        t.logpdf(obs[1], df=self.v,
                                 scale=self.scale, loc=loc[1])
                    log_likelihood += soft_assignment[obs_i, pred_j] * log_prob

            weights[i] = log_likelihood

        max_log_likelihood = np.max(weights)
        unnormalized_weights = np.exp(weights - max_log_likelihood)
        return unnormalized_weights / np.sum(unnormalized_weights)

if __name__ == "__main__":
    import matplotlib.pyplot as plt

    ball_num = 3
    state = np.random.rand(2, 4, ball_num)*100

    ideal_observ = BallObservation()

    print(f"True states:\n {state}")

    ideal_obs = ideal_observ.observe(state)
    print(f"Ideal observation:\n{ideal_obs}")

    print("--------------with onise---------------------")

    # print(state)
    observe_model = NormalObservation(ball_num)
    noisy_observe = observe_model.observe(state)
    print(state.shape, noisy_observe.shape)
    print(f"Noise observe:\n {noisy_observe}")

    print(f"Diff: {np.mean((ideal_obs-noisy_observe)**2)}")

    print(noisy_observe[0].shape)

    print(state.shape)
    print(state[:, :, 0:1].shape)

    # exit(0)

    probs = observe_model.evaluation(noisy_observe[0], state)
    print(f"final probs shape:{probs.shape}")

    fig, ax = plt.subplots(figsize=(12, 12))

    plot_observations(ax, state,
                      noisy_observe, observe_model.R)

    # plot_observations(ax, state[:, :, 1],
    #                   noisy_observe[:, :, 1], observe_model.R[2:4, 2:4])

    # plot_observations(ax, state[:, :, 2],
    #                   noisy_observe[:, :, 2], observe_model.R[4:6, 4:6])
    plt.show()
