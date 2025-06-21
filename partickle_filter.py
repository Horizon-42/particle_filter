import numpy as np
from transition_models import NormalTransition
from observision_models import NormalObservation
from math_utils import random_cov, random_diagonal_cov, sample_points_in_circle

class ParticleFilter:
    """
    Condensation Algorithm 
    """

    def __init__(self, particle_num: int, init_state: np.ndarray = None, state_dim: int = 4):
        self.N = particle_num

        # use gaussian to init particles
        x = np.random.uniform(-100, 100)
        y = np.random.uniform(0, 200)
        vx = np.random.uniform(-50, 50)
        vy = np.random.uniform(-50, 50)

        # sigma = random_diagonal_cov(4, 100000)
        # print(sigma)

        # particles = np.random.multivariate_normal(
        #     [x, y, vx, vy], sigma, size=self.N).reshape(-1, 4, 1)
        # print(particles.shape)

        xys = sample_points_in_circle((x, y), 2000, self.N)
        vs = sample_points_in_circle((vx, vy), 30, self.N)
        particles = np.concatenate([xys, vs], 1).reshape(self.N, state_dim, 1)

        weights = np.ones(self.N) / self.N  # uniform weights
        self.snaps = [(particles, weights)]

    def systematic_resample(self, particles, weights):
        """
        Systematic resampling of particles based on their weights.
        Returns indices of selected particles.
        """
        N = particles.shape[0]  # Number of particles

        # Normalize weights if not already done (assuming they are already normalized by update function)
        # weights /= np.sum(weights)

        # Compute cumulative sum of weights
        cumulative_sum = np.cumsum(weights)

        # points = np.random.rand(N)
        # Generate a starting point
        u0 = np.random.uniform(0, 1/N)
        # Generate N evenly spaced points
        points = u0 + np.arange(N) / N

        # Find the indices of the particles to be selected
        # This is a highly efficient way to do it using numpy broadcasting and searchsorted
        indices = np.searchsorted(cumulative_sum, points)

        return particles[indices]  # Select particles using the found indices

    def update(self, particles: np.ndarray, weights: np.ndarray, observation: np.ndarray):
        print(f"Neff:{1/np.sum(weights**2)}")
        print(
            f"weighs max:{np.max(weights)}, min:{np.min(weights)}, mean:{np.mean(weights)}")
        # sample from st-1
        new_particles = self.systematic_resample(particles, weights)

        # propagate the particles
        new_particles = NormalTransition.noisy_propagate(new_particles)
        # print(new_particles[:10])

        new_weights = NormalObservation.log_evaluation(
            observation, new_particles)

        self.snaps.append((new_particles, new_weights))
        return new_particles, new_weights
