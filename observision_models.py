import numpy as np
from math_utils import random_diagonal_cov, multivariate_normal_pdf_vectorized, multivariate_normal_logpdf_vectorized
from scipy.stats import multivariate_normal
from utils import plot_observations


class BallObservation:
    def __init__(self):
        self.C = np.array([[1, 0, 0, 0],
                  [0, 1, 0, 0]], dtype=float)

    def observe(self, state: np.ndarray):
        return self.C@state


class NormalObservation(BallObservation):
    def __init__(self, ball_num: int, var_scale: float = 1000):
        super().__init__()

        self.d_observe = ball_num*2
        self.R = random_diagonal_cov(self.d_observe, var_scale)

        self.noise_distribution = multivariate_normal(
            np.zeros(self.d_observe), self.R)

    def observe(self, state: np.ndarray):
        particle_num = state.shape[0]
        ball_num = state.shape[-1]
        noise = self.noise_distribution.rvs(size=particle_num)
        print(f"noise shape:{noise.shape}")
        return super().observe(state)
        # + noise.reshape(particle_num, 2, ball_num)

    def evaluation(self, single_observe: np.ndarray, states: np.ndarray):
        """
        Evaluates the likelihood of an observation given multiple particle states.

        Args:
            observe (np.ndarray): The current 2D observation (e.g., [z_x, z_y]).
                                  Should be a 1D or 2D array (e.g., (2,) or (1, 2)).
            states (np.ndarray): A 3D array of particle states (e.g., (N, 4, 1) if states are column vectors,
                                 or (N, 4) if states are row vectors).

        Returns:
            np.array: An array of likelihoods (unnormalized weights) for each particle.
        """
        # This assumes states are (N, D_state, 1) or (N, D_state)
        # If states are (N, D_state, 1) -> (N, D_obs, 1)
        expected_observations = self.C @ states
        expected_observations = expected_observations.reshape(
            -1, expected_observations.shape[1])

        # Ensure observe is 1D for pdf function
        weights = multivariate_normal_pdf_vectorized(
            single_observe.flatten(), expected_observations, self.R)
        return weights/np.sum(weights)


    def log_evaluation(self, single_observe: np.ndarray, states: np.ndarray):
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
            -1, self.d_observe, order='F')
        # print(expected_observations)
        # print(f"Expected observation shape:{expected_observations.shape}")
        # print(f"Single observation shape:{single_observe.shape}")
        # print(f"R shape:{self.R.shape}")

        # print(f"single observ:{single_observe}")
        # print(
        #     f"single observ reshaped:{single_observe.reshape(-1, self.d_observe, order='F').flatten()}")

        log_likelihoods = multivariate_normal_logpdf_vectorized(
            single_observe.flatten(), expected_observations, self.R)
        # print(
        #     f"loglikelihood max:{np.max(log_likelihoods)}, min{np.min(log_likelihoods)}, mean:{np.mean(log_likelihoods)}")
        # print(log_likelihoods)

        max_log_likelihood = np.max(log_likelihoods)

        # 将对数似然转换为非归一化权重，通过减去最大值避免exp(大正数)溢出
        # np.exp(log(w_i) - log(w_max)) = w_i / w_max
        unnormalized_weights = np.exp(log_likelihoods - max_log_likelihood)
        # print(f"exp unnormalized_weights:{unnormalized_weights}")

        return unnormalized_weights / np.sum(unnormalized_weights)




if __name__ == "__main__":
    import matplotlib.pyplot as plt

    ball_num = 3
    state = np.random.rand(3, 4, ball_num)*100

    ideal_observ = BallObservation()

    print(f"True states:\n {state}")

    ideal_obs = ideal_observ.observe(state)
    print(f"Ideal observation:\n{ideal_obs}")

    print("--------------with onise---------------------")

    # print(state)
    observe_model = NormalObservation(ball_num)
    noisy_observe = observe_model.observe(state)
    print(f"Noise observe:\n {noisy_observe}")

    print(f"Diff: {np.mean((ideal_obs-noisy_observe)**2)}")

    probs = observe_model.log_evaluation(noisy_observe[0], state)
    print(probs)

    fig, ax = plt.subplots(figsize=(12, 12))

    plot_observations(ax, state[:, :, 0],
                      noisy_observe[:, :, 0], observe_model.R[:2, :2])

    plot_observations(ax, state[:, :, 1],
                      noisy_observe[:, :, 1], observe_model.R[2:4, 2:4])

    # plot_observations(ax, state[:, :, 2],
    #                   noisy_observe[:, :, 2], observe_model.R[4:6, 4:6])
    plt.show()
