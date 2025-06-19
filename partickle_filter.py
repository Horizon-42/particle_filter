import numpy as np


class NormalTransition:
    """
    Qt = At@Qt-1 + Bt@action_t-1 + Noise(Nomal Distributed)
    """
    delta_t = 0.5
    A = np.array([[1, 0, delta_t, 0],
                  [0, 1, 0, delta_t],
                  [0, 0, 1, 0],
                  [0, 0, 0, 1]], dtype=float)

class ParticleFilter:
    """
    Condensation Algorithm 
    """

    def __init__(self, particle_num: int):
        self.N = particle_num
        self.particles = np.random.rand(self.N, 4)
        self.particles[:, 0] *= 3000 # x position in [0, 3000]
        self.particles[:, 1] *= 3000 # y position in [0, 3000]
        self.particles[:, 2] *= 200 # velocity in [0, 200]
        self.particles[:, 3] *= 200 # velocity in [0, 200]

        self.weights = np.ones(self.N) / self.N  # uniform weights
        self.snaps = [(self.particles, self.weights)]
    
    def __build_cumulative_range(self, weights:np.ndarray):
        cumul_range = np.zeros_like(weights)
        cumul_range[0] = weights[0]
        for i in range(1, cumul_range.shape[0]):
            cumul_range[i] = cumul_range[i-1]+weights[i]
        return cumul_range
    
    def update(self, particles:np.ndarray, weights:np.ndarray, observations:np.ndarray):
        # sample from st-1
        cumul_range = self.__build_cumulative_range(weights)
        new_particles = np.zeros_like(particles)
        # create N random numbers
        gen_rands = np.random.rand(self.N)

        for i in range(self.N):
            g_rand = gen_rands[i]
            n_i = 0
            for j in range(self.N):
                if g_rand <= cumul_range[j]:
                    n_i = j
                else:
                    break
            new_particles[i, :] = particles[n_i, :]
