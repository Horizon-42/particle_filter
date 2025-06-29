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

def normalize_log_likelihoods(log_likelihoods: np.ndarray):

    max_log_likelihood = np.max(log_likelihoods)

    # 将对数似然转换为非归一化权重，通过减去最大值避免exp(大正数)溢出
    # np.exp(log(w_i) - log(w_max)) = w_i / w_max
    unnormalized_weights = np.exp(log_likelihoods - max_log_likelihood)
    # print(f"exp unnormalized_weights:{unnormalized_weights}")

    return unnormalized_weights / np.sum(unnormalized_weights)

class BallObservation:
    def __init__(self, ball_num: int):
        self.C = np.array([[1, 0, 0, 0],
                  [0, 1, 0, 0]], dtype=float)
        self.ball_num = ball_num
        self.observ_orders = list(permutations(range(self.ball_num)))
        self.orders_num = len(self.observ_orders)

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


class UnorderedStudentTObservation(BallObservation):
    """
    Student's t-distribution based observation model for unordered observations.
    Assumes independent noise in x and y dimensions.
    """

    def __init__(self, ball_num: int, v: float = 0.5, scale: float = 10):
        super().__init__(ball_num)
        # Degrees of freedom (v or df). Lower v means fatter tails (more robust to outliers).
        self.v = v
        # Scale parameter. For a 2D independent model, this acts like std dev.
        self.scale = scale

        # For noise generation, we need to consider 2 dimensions.
        # rvs for a scalar df and scale means it samples from 1D t-dist.
        # We assume independent samples for x and y.
        # It's not a true multivariate t-distribution object, just for rvs convenience.

    def observe(self, state: np.ndarray):
        particle_num = state.shape[0]
        # Generate independent t-distributed noise for x and y for each ball and particle
        # size=particle_num * 2 * self.ball_num ensures enough samples for all (particle, dim, ball)
        noise = t.rvs(df=self.v, scale=self.scale,
                      size=particle_num * 2 * self.ball_num).reshape(particle_num, 2, self.ball_num)
        return super().observe(state) + noise

    def evaluation(self, single_observe: np.ndarray, states: np.ndarray):
        N_particles = states.shape[0]  # N

        # Particle predictions: shape (N_particles, 2, 1)
        predicted_positions = super().observe(states)

        log_phi = -np.log(self.ball_num)
        component_log_likelihoods = np.zeros(
            shape=(N_particles, self.ball_num))

        for obs_idx in range(self.ball_num):
            obx = single_observe[0, obs_idx]
            oby = single_observe[1, obs_idx]

            log_x_likelihoods = t.logpdf(
                obx, df=self.v, scale=self.scale, loc=predicted_positions[:, 0, 0])
            log_y_likelihoods = t.logpdf(
                oby, df=self.v, scale=self.scale, loc=predicted_positions[:, 1, 0])
            # assume x and y has independent noise
            component_log_likelihoods[:,
                                      obs_idx] = log_x_likelihoods+log_y_likelihoods

        log_likelihoods = logsumexp(log_phi+component_log_likelihoods, axis=1)

        # --- Normalize Weights using Log-Space Normalization ---
        max_log_likelihood = np.max(log_likelihoods)

        # Handle cases where all log-likelihoods are effectively negative infinity
        if np.isneginf(max_log_likelihood):
            # Default to uniform weights
            return np.ones(N_particles) / N_particles

        # Compute unnormalized weights: exp(log(w_i) - log(w_max)) = w_i / w_max
        unnormalized_weights = np.exp(
            log_likelihoods - max_log_likelihood)

        # Normalize the weights so they sum to 1
        sum_unnormalized_weights = np.sum(unnormalized_weights)
        if sum_unnormalized_weights == 0:
            # If all unnormalized weights are 0 (e.g., due to extreme underflow),
            # return uniform weights to prevent filter collapse.
            return np.ones(N_particles) / N_particles
        else:
            normalized_weights = unnormalized_weights / sum_unnormalized_weights

        return normalized_weights


class NearestStudentTObservation(BallObservation):
    """
    Student's t-distribution based observation model for unordered observations.
    Assumes independent noise in x and y dimensions.
    """

    def __init__(self, ball_num: int, v: float = 0.5, scale: float = 10):
        super().__init__(ball_num)
        # Degrees of freedom (v or df). Lower v means fatter tails (more robust to outliers).
        self.v = v
        # Scale parameter. For a 2D independent model, this acts like std dev.
        self.scale = scale

        # For noise generation, we need to consider 2 dimensions.
        # rvs for a scalar df and scale means it samples from 1D t-dist.
        # We assume independent samples for x and y.
        # It's not a true multivariate t-distribution object, just for rvs convenience.

    def observe(self, state: np.ndarray):
        particle_num = state.shape[0]
        # Generate independent t-distributed noise for x and y for each ball and particle
        # size=particle_num * 2 * self.ball_num ensures enough samples for all (particle, dim, ball)
        noise = t.rvs(df=self.v, scale=self.scale,
                      size=particle_num * 2 * self.ball_num).reshape(particle_num, 2, self.ball_num)
        return super().observe(state) + noise

    def evaluation(self, single_observe: np.ndarray, states: np.ndarray):
        N_particles = states.shape[0]  # N

        # Particle predictions: shape (N_particles, 2, 1)
        predicted_positions = super().observe(states)

        log_phi = -np.log(self.ball_num)
        component_log_likelihoods = np.zeros(
            shape=(N_particles, self.ball_num))

        for obs_idx in range(self.ball_num):
            obx = single_observe[0, obs_idx]
            oby = single_observe[1, obs_idx]

            log_x_likelihoods = t.logpdf(
                obx, df=self.v, scale=self.scale, loc=predicted_positions[:, 0, 0])
            log_y_likelihoods = t.logpdf(
                oby, df=self.v, scale=self.scale, loc=predicted_positions[:, 1, 0])
            # assume x and y has independent noise
            log_likelihoods_per_ob = log_x_likelihoods+log_y_likelihoods

            # normalize for each observs, so we can find out best fit
            component_log_likelihoods[:,
                                      obs_idx] = normalize_log_likelihoods(log_likelihoods_per_ob)

        # log_likelihoods = logsumexp(log_phi+component_log_likelihoods, axis=1)
        log_likelihoods = np.max(component_log_likelihoods, axis=1)

        # --- Normalize Weights using Log-Space Normalization ---
        max_log_likelihood = np.max(log_likelihoods)

        # Handle cases where all log-likelihoods are effectively negative infinity
        if np.isneginf(max_log_likelihood):
            # Default to uniform weights
            return np.ones(N_particles) / N_particles

        # Compute unnormalized weights: exp(log(w_i) - log(w_max)) = w_i / w_max
        unnormalized_weights = np.exp(
            log_likelihoods - max_log_likelihood)

        # Normalize the weights so they sum to 1
        sum_unnormalized_weights = np.sum(unnormalized_weights)
        if sum_unnormalized_weights == 0:
            # If all unnormalized weights are 0 (e.g., due to extreme underflow),
            # return uniform weights to prevent filter collapse.
            return np.ones(N_particles) / N_particles
        else:
            normalized_weights = unnormalized_weights / sum_unnormalized_weights

        return normalized_weights


class GMMObservation(BallObservation):
    def __init__(self, ball_num, var_scale: float = 10):
        super().__init__(ball_num)
        self.R = np.eye(2)
        np.fill_diagonal(self.R, var_scale)

        self.noise_distribution = multivariate_normal(
            np.zeros(2), self.R)

    def observe(self, state: np.ndarray):
        particle_num = state.shape[0]
        noise = self.noise_distribution.rvs(size=particle_num*self.ball_num)
        return super().observe(state) + noise.reshape(particle_num, 2, self.ball_num)

    def evaluation(self, single_observe: np.ndarray, states: np.ndarray):
        N_particles = states.shape[0]  # Number of particles

        # predicted_positions shape: (N_particles, 2, 1)
        predicted_positions = super().observe(states)
        # GMM component weights (phi_k). Assuming equal weights for each predicted ball.
        # This is the prior probability of an observation coming from a specific ball.
        log_phi = np.log(1.0 / self.ball_num)
        component_log_likelihoods = np.zeros(
            shape=(N_particles, self.ball_num))

        for obs_idx in range(self.ball_num):
            component_log_likelihoods[:, obs_idx] = multivariate_normal_logpdf_vectorized(single_observe[:, obs_idx].flatten(),
                                                                                          means=predicted_positions[
                :, :, 0],
                cov=self.R)

        # # -------------------------------------------------------------
        # # Apply the Log-Sum-Exp trick to calculate log(sum_k (phi_k * N_k)) for each particle.
        # # This accounts for the uncertainty of which predicted ball corresponds to current_obs.

        # # Add log(phi_k) to each component's log-likelihood
        # # log_terms shape: (N_particles, self.ball_num)
        log_likelihoods = logsumexp(
            log_phi + component_log_likelihoods,
            axis=1
        )

        # Convert total log-likelihoods to normalized weights using log-space normalization.
        # This prevents numerical overflow when exponentiating large positive log-likelihoods.
        max_log_likelihood = np.max(log_likelihoods)

        # Handle cases where all total_log_likelihoods are effectively negative infinity
        # (meaning all particles are extremely unlikely given observations).
        if np.isneginf(max_log_likelihood):
            # Default to uniform weights
            return np.ones(N_particles) / N_particles

        # Compute unnormalized weights: exp(log(w_i) - log(w_max)) = w_i / w_max
        unnormalized_weights = np.exp(
            log_likelihoods - max_log_likelihood)

        # Normalize the weights so they sum to 1.
        sum_unnormalized_weights = np.sum(unnormalized_weights)
        if sum_unnormalized_weights == 0:
            # If all unnormalized weights are 0 (e.g., due to extreme underflow or poor initial guess),
            # return uniform weights to prevent filter collapse.
            return np.ones(N_particles) / N_particles
        else:
            normalized_weights = unnormalized_weights / sum_unnormalized_weights

        return normalized_weights


class NearestNormalObservation(BallObservation):  # Renamed class for clarity
    def __init__(self, ball_num: int, var_scale: float = 10):
        super().__init__(ball_num)
        self.R = np.eye(2)
        np.fill_diagonal(self.R, var_scale)

        self.R_inv = np.linalg.inv(self.R)
        self.det_R = np.linalg.det(self.R)
        self.k_dim = 2  # Dimension of each observation (x, y)

        self.noise_distribution = multivariate_normal(
            np.zeros(self.k_dim), self.R)

    def observe(self, state: np.ndarray):
        particle_num = state.shape[0]
        noise = self.noise_distribution.rvs(size=particle_num * self.ball_num)
        return super().observe(state) + noise.reshape(particle_num, self.k_dim, self.ball_num)

    def evaluation(self, single_observe: np.ndarray, states: np.ndarray):
        N_particles = states.shape[0]  # Number of particles

        # predicted_positions shape: (N_particles, 2, 1)
        predicted_positions = super().observe(states)

        component_log_likelihoods = np.zeros(
            shape=(N_particles, self.ball_num))

        for obs_idx in range(self.ball_num):
            component_log_likelihoods[:, obs_idx] = multivariate_normal_logpdf_vectorized(single_observe[:, obs_idx].flatten(),
                                                                                          means=predicted_positions[
                :, :, 0],
                cov=self.R)

        log_likelihoods = np.max(component_log_likelihoods, axis=1)

        # Convert total log-likelihoods to normalized weights using log-space normalization.
        # This prevents numerical overflow when exponentiating large positive log-likelihoods.
        max_log_likelihood = np.max(log_likelihoods)

        # Handle cases where all total_log_likelihoods are effectively negative infinity
        # (meaning all particles are extremely unlikely given observations).
        if np.isneginf(max_log_likelihood):
            # Default to uniform weights
            return np.ones(N_particles) / N_particles

        # Compute unnormalized weights: exp(log(w_i) - log(w_max)) = w_i / w_max
        unnormalized_weights = np.exp(
            log_likelihoods - max_log_likelihood)

        # Normalize the weights so they sum to 1.
        sum_unnormalized_weights = np.sum(unnormalized_weights)
        if sum_unnormalized_weights == 0:
            # If all unnormalized weights are 0 (e.g., due to extreme underflow or poor initial guess),
            # return uniform weights to prevent filter collapse.
            return np.ones(N_particles) / N_particles
        else:
            normalized_weights = unnormalized_weights / sum_unnormalized_weights

        return normalized_weights

if __name__ == "__main__":
    import matplotlib.pyplot as plt

    ball_num = 3
    state = np.random.rand(12, 4, ball_num)*100

    ideal_observ = BallObservation(ball_num)

    print(f"True states:\n {state}")

    ideal_obs = ideal_observ.observe(state)
    print(f"Ideal observation:\n{ideal_obs}")

    print("--------------with onise---------------------")

    # print(state)
    observe_model = GMMObservation(ball_num=ball_num)
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
