import numpy as np


class ParticleFilter:
    """
    Condensation Algorithm 
    """
    def __init__(self, partickle_num:int):
        self.particles = np.random.rand(partickle_num, 4)
        self.particles[:, 0] *= 3000 # x position in [0, 3000]
        self.particles[:, 1] *= 3000 # y position in [0, 3000]
        self.particles[:, 2] *= 200 # velocity in [0, 200]
        self.particles[:, 3] *= 200 # velocity in [0, 200]

        self.weights = np.ones(partickle_num) / partickle_num  # uniform weights

        self.snaps = [(self.particles, self.weights)]
    
    def __build_cumulative_range(self, weights:np.ndarray):
        cumul_range = np.zeros_like(weights)
        cumul_range[0] = weights[0]
        for i in range(1, self.__cum_range.shape[0]):
            cumul_range[i] = cumul_range[i-1]+weights[i]
        return cumul_range
    
    def update(self, particles:np.ndarray, weights:np.ndarray, observations:np.ndarray):
        pass