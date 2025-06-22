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
    def __init__(self):
        super().__init__()

        self.R = random_diagonal_cov(2, 10000)

        self.noise_distribution = multivariate_normal([0]*2, self.R)

    def observe(self, state: np.ndarray):
        return super().observe(state) + self.noise_distribution.rvs().reshape(-1, 1)
        return super().observe(state)

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
        expected_observations = self.C @ states
        expected_observations = expected_observations.reshape(
            -1, expected_observations.shape[1])

        log_likelihoods = multivariate_normal_logpdf_vectorized(
            single_observe.flatten(), expected_observations, self.R)
        # print(
        #     f"loglikelihood max:{np.max(log_likelihoods)}, min{np.min(log_likelihoods)}, mean:{np.mean(log_likelihoods)}")

        max_log_likelihood = np.max(log_likelihoods)

        # 将对数似然转换为非归一化权重，通过减去最大值避免exp(大正数)溢出
        # np.exp(log(w_i) - log(w_max)) = w_i / w_max
        unnormalized_weights = np.exp(log_likelihoods - max_log_likelihood)

        return unnormalized_weights / np.sum(unnormalized_weights)




if __name__ == "__main__":
    import matplotlib.pyplot as plt

    state = np.random.rand(2, 4, 1)*100
    print(state)
    # print(state)
    observe_model = NormalObservation()
    observe = observe_model.observe(state)
    print(observe)

    for i in range(observe.shape[0]):
        print(state[i, :2] == observe[i])

    # probs = [NormalObservation.evaluation(
    #     observe[i], state[i, :]) for i in range(observe.shape[0])]
    probs = observe_model.evaluation(observe[0], state)
    print(probs)

    fig, ax = plt.subplots(figsize=(12, 12))

    plot_observations(ax, state, observe, observe_model.R)
    plt.show()
