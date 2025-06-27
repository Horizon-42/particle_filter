import numpy as np
from transition_models import BallTransition, NormalTransition, UniformTransition, StudentTTransition, TransitionType
from observision_models import NormalObservation, BallObservation, StudentTObservation
from math_utils import random_cov, random_diagonal_cov, sample_points_in_circle



class ParticleFilter:
    """
    Condensation Algorithm 
    """

    def __init__(self, delta_t: float, particle_num: int, ball_num: int,
                 transition_type: TransitionType, observ_model: BallObservation):
        self.N = particle_num

        self.trans_model: BallTransition = None
        if transition_type == TransitionType.Normal:
            self.trans_model = NormalTransition(delta_t=delta_t)
        elif transition_type == TransitionType.Uniform:
            self.trans_model = UniformTransition(delta_t=delta_t)
        elif transition_type == TransitionType.StudentT:
            self.trans_model = StudentTTransition(delta_t=delta_t)
        else:
            self.trans_model = NormalTransition(delta_t=delta_t)

        self.observe_model: BallObservation = observ_model

        # use gaussian to init particles
        x = np.random.uniform(0, 4000)
        y = np.random.uniform(0, 4000)
        vx = np.random.uniform(0, 200)
        vy = np.random.uniform(0, 200)

        self.init_particles = np.zeros(shape=(particle_num, 4, ball_num))

        for i in range(0, ball_num):
            # self.init_particles[:, :, i] = self.init_particles[:, :, 0]
            xys = sample_points_in_circle((x, y), 10000, particle_num)
            vs = sample_points_in_circle((vx, vy), 200, particle_num)

            self.init_particles[:, :, i] = np.concatenate(
                [xys, vs], 1).reshape(particle_num, 4)


        self.init_weights = np.ones(self.N) / self.N  # uniform weights

    def systematic_resample(self, particles: np.ndarray, weights: np.ndarray):
        """
        Systematic resampling of particles based on their weights.
        Returns indices of selected particles.
        """
        N = particles.shape[0]  # Number of particles

        # Normalize weights if not already done (assuming they are already normalized by update function)
        # weights /= np.sum(weights)

        # Compute cumulative sum of weights
        cumulative_sum = np.cumsum(weights)

        # Generate a starting point
        u0 = np.random.uniform(0, 1/N)
        # # Generate N evenly spaced points
        points = u0 + np.arange(N) / N

        # Find the indices of the particles to be selected
        # This is a highly efficient way to do it using numpy broadcasting and searchsorted
        indices = np.searchsorted(cumulative_sum, points)

        return particles[indices]  # Select particles using the found indices

    def multinomial_resample(self, particles: np.ndarray, weights: np.ndarray):
        """
        Systematic resampling of particles based on their weights.
        Returns indices of selected particles.
        """
        N = particles.shape[0]  # Number of particles

        # Normalize weights if not already done (assuming they are already normalized by update function)
        # weights /= np.sum(weights)

        # Compute cumulative sum of weights
        cumulative_sum = np.cumsum(weights)

        # multinomial
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
        new_particles = self.multinomial_resample(particles, weights)

        # propagate the particles
        new_particles = self.trans_model.propagate(new_particles)
        # print(new_particles[:10])

        if observation is not None:
            new_weights = self.observe_model.evaluation(
                observation, new_particles)
        else:
            new_weights = np.ones(self.N) / self.N

        return new_particles, new_weights
