import numpy as np
from transition_models import NormalTransition
from observision_models import NormalObservation

class ParticleFilter:
    """
    Condensation Algorithm 
    """

    def __init__(self, particle_num: int):
        self.N = particle_num
        self.particles = np.random.rand(self.N, 4, 1)
        self.particles[:, 0] *= 300  # x position in [0, 3000]
        self.particles[:, 1] *= 300  # y position in [0, 3000]
        self.particles[:, 2] *= 2000  # velocity in [0, 200]
        self.particles[:, 3] *= 2000  # velocity in [0, 200]

        self.weights = np.ones(self.N) / self.N  # uniform weights
        self.snaps = [(self.particles, self.weights)]
    
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

        # # Generate N evenly spaced points, shifted by a random offset
        # # These are the "pointers" into the cumulative sum
        # # The first random number is between 0 and 1/N
        # r0 = np.random.uniform(0, 1/N)
        # points = np.arange(N) / N + r0  # [r0, r0 + 1/N, ..., r0 + (N-1)/N]
        points = np.random.rand(N)

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
