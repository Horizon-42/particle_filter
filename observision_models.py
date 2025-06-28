import numpy as np
from math_utils import random_diagonal_cov, multivariate_normal_pdf_vectorized, multivariate_normal_logpdf_vectorized
from scipy.stats import multivariate_normal
from utils import plot_observations
from scipy.stats import t
from itertools import product
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
        self.pairs = list(product(range(self.ball_num), range(self.ball_num)))
        self.pairs_num = len(self.pairs)
        print(self.pairs_num)

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


class OrderedStudentTObservation(BallObservation):
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

    def evaluation(self, single_observe: np.ndarray, states: np.ndarray):
        N = states.shape[0]
        expected_observations = super().observe(states).reshape(N, -1, order='F')

        observs = single_observe.reshape(-1, order='F')
        log_likelihoods = t.logpdf(
            observs, df=self.v, scale=self.scale, loc=expected_observations)
        # TODO why mean is better? it should be sum
        log_likelihoods = np.mean(log_likelihoods, axis=1)

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
        N_predicted_balls = self.ball_num    # B
        # O (can be different from B)
        N_observed_points = single_observe.shape[-1]

        # Particle predictions: shape (N_particles, 2, N_predicted_balls)
        predicted_positions = super().observe(states)

        # Initialize total log-likelihoods for each particle
        total_log_likelihoods = np.zeros(N_particles)

        # Log-prior for GMM components (assuming uniform: 1/B)
        # This will be added to each component's log-likelihood before logsumexp
        log_prior_component = -np.log(self.ball_num)

        log_likelihood_components = np.zeros(
            shape=(N_particles, N_observed_points, N_predicted_balls))
        # Iterate through each actual observed point
        for obs_idx in range(N_observed_points):
            current_obs = single_observe[:, obs_idx]  # Shape: (2,)

            # For each particle and each predicted ball, calculate the log-likelihood
            # log_likelihood_components shape: (N_particles, N_predicted_balls)
            # This will store log P(current_obs | predicted_ball_k_for_particle_i)

            # Extract x and y components of predicted positions
            # pred_x_coords shape: (N_particles, N_predicted_balls)
            pred_x_coords = predicted_positions[:, 0, :]
            pred_y_coords = predicted_positions[:, 1, :]

            # Calculate log-PDF for x and y dimensions independently
            # logpdf_x_comp shape: (N_particles, N_predicted_balls)
            logpdf_x_comp = t.logpdf(
                current_obs[0], df=self.v, scale=self.scale, loc=pred_x_coords)
            # logpdf_y_comp shape: (N_particles, N_predicted_balls)
            logpdf_y_comp = t.logpdf(
                current_obs[1], df=self.v, scale=self.scale, loc=pred_y_coords)

            # Combine x and y log-PDFs to get the 2D joint log-PDF for each component
            # (Assuming independence of x and y dimensions, which is implied by your scale usage)
            log_likelihood_components[:, obs_idx, :] = normalize_log_likelihoods(logpdf_x_comp +
                                                                                 # (N_particles, N_predicted_balls)
                                                                                 logpdf_y_comp)

        log_likelihood_components = log_likelihood_components.reshape(
            N_particles, -1)
        print(
            f"log_likelihood_components shape:{log_likelihood_components.shape}")

        # log_likelihood_components[:] = normalize_log_likelihoods(
        #     log_likelihood_components[:])
        total_log_likelihoods = logsumexp(
            log_prior_component+log_likelihood_components, axis=1)

        # total_log_likelihoods = np.max(log_likelihood_components, axis=1)

        # --- Normalize Weights using Log-Space Normalization ---
        max_log_likelihood = np.max(total_log_likelihoods)

        # Handle cases where all log-likelihoods are effectively negative infinity
        if np.isneginf(max_log_likelihood):
            # Default to uniform weights
            return np.ones(N_particles) / N_particles

        # Compute unnormalized weights: exp(log(w_i) - log(w_max)) = w_i / w_max
        unnormalized_weights = np.exp(
            total_log_likelihoods - max_log_likelihood)

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
        N_predicted_balls = self.ball_num    # B
        # O (can be different from B)
        N_observed_points = single_observe.shape[-1]

        # Particle predictions: shape (N_particles, 2, N_predicted_balls)
        predicted_positions = super().observe(states)
        print(predicted_positions.shape)

        # Initialize total log-likelihoods for each particle
        total_log_likelihoods = np.zeros(N_particles)

        # log likelihoods for all particles on every observes
        # shape: (N_particles, N_predicted_balls, N_observed_points)
        log_likelihood_components = np.zeros(
            shape=(N_particles, N_predicted_balls, N_observed_points))

        # Iterate through each actual observed point
        for obs_idx in range(N_observed_points):
            current_obs = single_observe[:, obs_idx]  # Shape: (2,)

            # For each particle and each predicted ball, calculate the log-likelihood
            # log_likelihood_components shape: (N_particles, N_predicted_balls)
            # This will store log P(current_obs | predicted_ball_k_for_particle_i)

            # Extract x and y components of predicted positions
            # pred_x_coords shape: (N_particles, N_predicted_balls)
            pred_x_coords = predicted_positions[:, 0, :]
            pred_y_coords = predicted_positions[:, 1, :]

            # Calculate log-PDF for x and y dimensions independently
            # logpdf_x_comp shape: (N_particles, N_predicted_balls)
            logpdf_x_comp = t.logpdf(
                current_obs[0], df=self.v, scale=self.scale, loc=pred_x_coords)
            # logpdf_y_comp shape: (N_particles, N_predicted_balls)
            logpdf_y_comp = t.logpdf(
                current_obs[1], df=self.v, scale=self.scale, loc=pred_y_coords)

            # Combine x and y log-PDFs to get the 2D joint log-PDF for each component
            # (Assuming independence of x and y dimensions, which is implied by your scale usage)
            # Shape(N_particles, N_preducted_balls)
            log_likelihoods_per_obs = logpdf_x_comp + logpdf_y_comp
            # exp convert to parobs
            likelihoods_per_obs = np.exp(log_likelihoods_per_obs)
            # normalize for the obs
            likelihoods_per_obs[:] /= np.sum(likelihoods_per_obs[:])
            log_likelihood_components[:, :, obs_idx] = np.log(
                likelihoods_per_obs)

        # Sum likelihoods for each ball in all particles
        total_log_likelihoods_per_balls = np.sum(
            log_likelihood_components, axis=0)

        # total_log_likelihoods = np.sum(
        #     log_likelihood_components[:,:,], axis=1)
        loglikelihoods_per_balls = np.zeros(
            shape=(N_particles, N_predicted_balls))
        for i in range(N_predicted_balls):
            loglikelihoods_per_balls[:, i] = log_likelihood_components[:, i, np.argmax(
                total_log_likelihoods_per_balls[i, :])]
        total_log_likelihoods = np.mean(
            loglikelihoods_per_balls, axis=1)

        # --- Normalize Weights using Log-Space Normalization ---
        max_log_likelihood = np.max(total_log_likelihoods)

        # Handle cases where all log-likelihoods are effectively negative infinity
        if np.isneginf(max_log_likelihood):
            # Default to uniform weights
            return np.ones(N_particles) / N_particles

        # Compute unnormalized weights: exp(log(w_i) - log(w_max)) = w_i / w_max
        unnormalized_weights = np.exp(
            total_log_likelihoods - max_log_likelihood)

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

        # predicted_positions shape: (N_particles, 2, ball_num)
        predicted_positions = super().observe(states)

        # single_observe shape: (2, ball_num_observed)
        # Transpose to (ball_num_observed, 2) for easier iteration over observations
        single_observe_T = single_observe.T  # shape: (ball_num_observed, 2)
        # Actual number of observed points
        ball_num_observed = single_observe_T.shape[0]

        # Initialize total log-likelihoods for each particle
        # These will be accumulated for each observed point
        total_log_likelihoods = np.zeros(N_particles)

        # GMM component weights (phi_k). Assuming equal weights for each predicted ball.
        # This is the prior probability of an observation coming from a specific ball.
        log_phi = np.log(1.0 / self.ball_num)

        # Iterate through each actual observed point (o_x, o_y)
        # TODO 计算所有可能分布的联合似然
        for obs_idx in range(ball_num_observed):
            current_obs = single_observe_T[obs_idx, :]  # Shape: (2,)

            # component_log_likelihoods will store the log-likelihood of current_obs
            # given each predicted ball's position, for all particles.
            # Shape: (N_particles, self.ball_num)
            component_log_likelihoods = np.zeros((N_particles, self.ball_num))

            for ball_idx in range(self.ball_num):
                # predicted_pos_for_ball_idx shape: (N_particles, 2)
                # These are the means (mu_k) for the GMM components for each particle.
                predicted_pos_for_ball_idx = predicted_positions[:, :, ball_idx]

                logpdf = multivariate_normal_logpdf_vectorized(current_obs.flatten(),
                                                               means=predicted_pos_for_ball_idx,
                                                               cov=self.R)
                component_log_likelihoods[:, ball_idx] = logpdf

            # -------------------------------------------------------------
            # Apply the Log-Sum-Exp trick to calculate log(sum_k (phi_k * N_k)) for each particle.
            # This accounts for the uncertainty of which predicted ball corresponds to current_obs.

            # Add log(phi_k) to each component's log-likelihood
            # log_terms shape: (N_particles, self.ball_num)
            log_sum_exp_for_obs = logsumexp(
                log_phi + component_log_likelihoods,
                axis=1
            )

            # Accumulate total log-likelihoods for each particle.
            # Since observations are assumed conditionally independent given the state,
            # the total log-likelihood is the sum of individual log-likelihoods.
            total_log_likelihoods += log_sum_exp_for_obs
            # -------------------------------------------------------------

        # Convert total log-likelihoods to normalized weights using log-space normalization.
        # This prevents numerical overflow when exponentiating large positive log-likelihoods.
        max_total_log_likelihood = np.max(total_log_likelihoods)

        # Handle cases where all total_log_likelihoods are effectively negative infinity
        # (meaning all particles are extremely unlikely given observations).
        if np.isneginf(max_total_log_likelihood):
            # Default to uniform weights
            return np.ones(N_particles) / N_particles

        # Compute unnormalized weights: exp(log(w_i) - log(w_max)) = w_i / w_max
        unnormalized_weights = np.exp(
            total_log_likelihoods - max_total_log_likelihood)

        # Normalize the weights so they sum to 1.
        sum_unnormalized_weights = np.sum(unnormalized_weights)
        if sum_unnormalized_weights == 0:
            # If all unnormalized weights are 0 (e.g., due to extreme underflow or poor initial guess),
            # return uniform weights to prevent filter collapse.
            return np.ones(N_particles) / N_particles
        else:
            normalized_weights = unnormalized_weights / sum_unnormalized_weights

        return normalized_weights


class NearestNeighborObservation(BallObservation):  # Renamed class for clarity
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

        # predicted_positions shape: (N_particles, 2, ball_num)
        predicted_positions = super().observe(states)

        # single_observe shape: (2, ball_num_observed)
        # Transpose to (ball_num_observed, 2) for easier iteration over observations
        single_observe_T = single_observe.T  # shape: (ball_num_observed, 2)
        # Actual number of observed points
        ball_num_observed = single_observe_T.shape[0]

        # Initialize total log-likelihoods for each particle
        # These will be accumulated for each observed point
        total_log_likelihoods = np.zeros(N_particles)

        # Handle case with no observations:
        if ball_num_observed == 0:
            return np.ones(N_particles) / N_particles  # Return uniform weights

        # Check for scenarios with no predicted balls
        if self.ball_num == 0:
            # If no predicted balls, and there are observations, this scenario is problematic
            # All particles effectively have zero likelihood if there's nothing to match.
            # Returning uniform weights as a safe fallback, or a small epsilon likelihood.
            return np.ones(N_particles) / N_particles

        # Iterate through each actual observed point (o_x, o_y)
        for obs_idx in range(ball_num_observed):
            current_obs = single_observe_T[obs_idx, :]  # Shape: (2,)

            # component_log_likelihoods will store the log-likelihood of current_obs
            # given each predicted ball's position, for all particles.
            # Shape: (N_particles, self.ball_num)
            component_log_likelihoods = np.zeros((N_particles, self.ball_num))

            for ball_idx in range(self.ball_num):
                # predicted_pos_for_ball_idx shape: (N_particles, 2)
                # These are the means (mu_k) for the GMM components for each particle.
                predicted_pos_for_ball_idx = predicted_positions[:, :, ball_idx]

                # Using the custom vectorized logpdf function
                logpdf = multivariate_normal_logpdf_vectorized(current_obs.flatten(),
                                                               means=predicted_pos_for_ball_idx,
                                                               cov=self.R)
                component_log_likelihoods[:, ball_idx] = logpdf

            # -------------------------------------------------------------
            # KEY CHANGE FOR NEAREST NEIGHBOR (NN) / MAX LIKELIHOOD ASSOCIATION
            # We take the maximum log-likelihood for each observation across all predicted balls.
            # This represents the "best match" for the current observation for each particle.

            # log_likelihood_for_obs shape: (N_particles,)
            # This is log( p(current_obs | particle_state, best_association_for_obs) )
            log_likelihood_for_obs = np.max(component_log_likelihoods, axis=1)

            # Accumulate total log-likelihoods for each particle.
            # Since observations are assumed conditionally independent given the state,
            # the total log-likelihood is the sum of individual log-likelihoods.
            total_log_likelihoods += log_likelihood_for_obs
            # -------------------------------------------------------------

        # Convert total log-likelihoods to normalized weights using log-space normalization.
        max_total_log_likelihood = np.max(total_log_likelihoods)

        # Handle cases where all total_log_likelihoods are effectively negative infinity
        if np.isneginf(max_total_log_likelihood):
            # Default to uniform weights
            return np.ones(N_particles) / N_particles

        unnormalized_weights = np.exp(
            total_log_likelihoods - max_total_log_likelihood)

        sum_unnormalized_weights = np.sum(unnormalized_weights)
        if sum_unnormalized_weights == 0:
            return np.ones(N_particles) / N_particles
        else:
            normalized_weights = unnormalized_weights / sum_unnormalized_weights

        return normalized_weights

if __name__ == "__main__":
    import matplotlib.pyplot as plt

    ball_num = 3
    state = np.random.rand(2, 4, ball_num)*100

    ideal_observ = BallObservation(ball_num)

    print(f"True states:\n {state}")

    ideal_obs = ideal_observ.observe(state)
    print(f"Ideal observation:\n{ideal_obs}")

    print("--------------with onise---------------------")

    # print(state)
    observe_model = UnorderedStudentTObservation(ball_num=ball_num)
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
