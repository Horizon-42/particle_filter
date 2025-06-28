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
                 speed_range: list[float] = [-100, 100]):
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

    def residual_resample(self, particles: np.ndarray, weights: np.ndarray):
        """
        对粒子权重进行残差重采样。

        参数:
        weights (numpy.ndarray): 一个包含所有粒子归一化权重的 NumPy 数组。
                                这些权重必须是非负数且总和接近 1。

        返回:
        numpy.ndarray: 一个包含 N 个整数索引的 NumPy 数组，表示重采样后粒子的新索引。
                    你可以使用这些索引来从旧的粒子集中构建新的粒子集。
        """

        N = len(weights)  # 粒子总数
        new_indices = np.zeros(N, dtype=int)  # 用于存储新粒子索引的数组

        # --- 1. 确定性复制部分 ---
        # 计算每个粒子期望被复制的次数 (N * w_i)
        expected_counts = N * weights

        # 确定性地复制整数部分
        # floor(expected_counts) 得到整数部分，例如 3.7 -> 3
        # astype(int) 将浮点数转换为整数
        num_copies_integer = np.floor(expected_counts).astype(int)

        current_idx = 0
        for i in range(N):
            # 将粒子 i 复制 num_copies_integer[i] 次
            for _ in range(num_copies_integer[i]):
                if current_idx < N:  # 确保不会超出 new_indices 的范围
                    new_indices[current_idx] = i
                    current_idx += 1
                else:
                    break  # 已经复制了 N 个粒子，提前退出

        # --- 2. 随机复制部分 ---
        # 计算剩余的粒子数量，这些粒子需要通过随机采样来补充
        num_remaining_particles = N - current_idx

        if num_remaining_particles > 0:
            # 计算每个粒子的“剩余权重”（小数部分）
            residual_weights = expected_counts - num_copies_integer

            # 对剩余权重进行归一化，以便进行多项式采样
            # 确保剩余权重之和不为零，避免除以零的错误
            sum_residual_weights = np.sum(residual_weights)
            if sum_residual_weights > 0:
                normalized_residual_weights = residual_weights / sum_residual_weights
            else:
                # 如果所有剩余权重都为零（即所有粒子都被整数次复制），
                # 意味着 current_idx 应该已经等于 N。
                # 如果走到这里，表示存在浮点数精度问题或逻辑错误，
                # 简单处理为随机选择剩余粒子（不推荐，但作为兜底）
                normalized_residual_weights = np.ones(N) / N  # 均匀分布

            # 使用 numpy.random.choice 进行多项式采样
            # p 参数必须是归一化的概率分布
            remaining_indices = np.random.choice(
                N,
                size=num_remaining_particles,
                p=normalized_residual_weights
            )

            # 将随机采样的粒子添加到新索引数组的剩余位置
            new_indices[current_idx:] = remaining_indices

        # 返回重采样后的粒子索引
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
        new_particles = self.residual_resample(particles, weights)

        # propagate the particles
        new_particles = self.trans_model.propagate(new_particles)
        # print(new_particles[:10])

        if observation is not None:
            new_weights = self.observe_model.evaluation(
                observation, new_particles)
        else:
            new_weights = np.ones(self.N) / self.N

        return new_particles, new_weights
