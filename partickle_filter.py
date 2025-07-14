import numpy as np
from transition_models import BallTransition, NormalTransition, UniformTransition, StudentTTransition, TransitionType
from observision_models import BallObservation
from math_utils import random_cov, random_diagonal_cov, sample_points_in_circle



class ParticleFilter:
    """
    Condensation Algorithm 
    """

    def __init__(self, delta_t: float, particle_num: int, ball_num: int,
                 transition_type: TransitionType, observ_model: BallObservation,
                 pos_range: list[float] = [-100, 100],
                 speed_range: list[float] = [0, 100]):
        self.N = particle_num

        # Select the transition model based on the type
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
        self.init_particles = np.zeros(shape=(particle_num, 4, ball_num))

        for i in range(0, ball_num):
            # self.init_particles[:, :, i] = self.init_particles[:, :, 0]
            xs = np.random.uniform(
                pos_range[0], pos_range[1], size=particle_num)
            ys = np.random.uniform(
                pos_range[0], pos_range[1], size=particle_num)
            vxs = np.random.uniform(
                speed_range[0], speed_range[1], size=particle_num)
            vys = np.random.uniform(
                speed_range[0], speed_range[1],  size=particle_num)

            self.init_particles[:, :, i] = np.vstack([xs, ys, vxs, vys]).T


        self.init_weights = np.ones(self.N) / self.N  # uniform weights

        self.ball_indices = np.random.randint(0, ball_num, size=self.N)

        self.resampled = False

    def residual_resample(self, particles: np.ndarray, weights: np.ndarray):
        """
        Perform residual resampling on particle weights.

        Parameters:
        weights (numpy.ndarray): A NumPy array containing the normalized weights of all particles.
                                 These weights must be non-negative and sum approximately to 1.

        Returns:
        numpy.ndarray: A NumPy array containing N integer indices, representing the new indices of particles after resampling.
                       You can use these indices to construct a new particle set from the old one.
        """

        N = len(weights)  # total number of particles
        new_indices = np.zeros(N, dtype=int)  # to store new indices

        # --- 1. Deterministic replication part ---
        # Compute the expected number of times each particle should be replicated (N * w_i)
        expected_counts = N * weights

        # copy the integer part of expected counts
        num_copies_integer = np.floor(expected_counts).astype(int)

        current_idx = 0
        for i in range(N):
            # Replicate each particle according to its integer expected count
            for _ in range(num_copies_integer[i]):
                if current_idx < N:  # 确保不会超出 new_indices 的范围
                    new_indices[current_idx] = i
                    current_idx += 1
                else:
                    break  # 已经复制了 N 个粒子，提前退出

        # --- 2. Random sampling part ---
        num_remaining_particles = N - current_idx

        if num_remaining_particles > 0:
            residual_weights = expected_counts - num_copies_integer

            sum_residual_weights = np.sum(residual_weights)
            if sum_residual_weights > 0:
                normalized_residual_weights = residual_weights / sum_residual_weights
            else:
                # if all residual weights are zero, somthing is wrong, use uniform distribution
                normalized_residual_weights = np.ones(N) / N

            # use np.random.choice to sample the remaining particles
            remaining_indices = np.random.choice(
                N,
                size=num_remaining_particles,
                p=normalized_residual_weights
            )

            # Fill the remaining indices in new_indices
            new_indices[current_idx:] = remaining_indices

        # Update ball indices based on selected particles
        self.ball_indices = self.ball_indices[new_indices]
        return particles[new_indices]


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

        # Update ball indices based on selected particles
        self.ball_indices = self.ball_indices[indices]

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
        # Update ball indices based on selected particles
        self.ball_indices = self.ball_indices[indices]

        return particles[indices]  # Select particles using the found indices

    def update(self, particles: np.ndarray, weights: np.ndarray, observation: np.ndarray):
        neff = 1.0 / np.sum(weights**2)
        print(f"Effective sample size: {neff}")

        new_particles = self.systematic_resample(particles, weights)
        # propagate the particles
        new_particles = self.trans_model.propagate(new_particles)
        # print(new_particles[:10])

        if observation is None:
            return new_particles, np.ones(self.N) / self.N

        # judge if recompute weights or not
        new_weights, self.ball_indices = self.observe_model.evaluation(
            observation, new_particles)

        return new_particles, new_weights
