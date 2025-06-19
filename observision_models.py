import numpy as np
from math_utils import random_diagonal_cov, multivariate_normal_pdf_vectorized
from scipy.stats import multivariate_normal
from utils import plot_observations

class NormalObservation:
    C = np.array([[1, 0, 0, 0],
                  [0, 1, 0, 0]], dtype=float)

    R = random_diagonal_cov(2, 1000)

    noise_distribution = multivariate_normal([0]*2, R)

    @classmethod
    def observe(cls, state: np.ndarray):
        observation = cls.C @ state
        # return observation
        return observation + cls.noise_distribution.rvs().reshape(-1, 1)


    @classmethod
    # Changed 'state' to 'states' (plural)
    def evaluation(cls, single_observe: np.ndarray, states: np.ndarray):
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
        expected_observations = cls.C @ states
        expected_observations = expected_observations.reshape(
            -1, expected_observations.shape[1])

        # Now, evaluate PDF for all particles at once.
        # x is the single observation, mean is the array of expected observations (one for each particle)
        # cov is the fixed observation noise covariance
        # The pdf function will broadcast 'x' to match the shape of 'mean' and 'cov'

        # Ensure observe is 1D for pdf function
        return multivariate_normal_pdf_vectorized(single_observe.flatten(), expected_observations, cls.R)


if __name__ == "__main__":
    import matplotlib.pyplot as plt

    state = np.random.rand(10, 4, 1)*100
    print(state)
    # print(state)
    observe = NormalObservation.observe(state)
    print(observe)

    for i in range(3):
        print(state[i, :2] == observe[i])

    # probs = [NormalObservation.evaluation(
    #     observe[i], state[i, :]) for i in range(observe.shape[0])]
    probs = NormalObservation.evaluation(observe[0], state)
    print(probs)

    fig, ax = plt.subplots(figsize=(12, 12))

    plot_observations(ax, state, observe, NormalObservation.R)
    plt.show()
