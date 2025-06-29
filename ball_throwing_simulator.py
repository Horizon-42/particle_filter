# simulate motion and observation
import numpy as np
from transition_models import NormalTransition, BallTransition, UniformTransition
from observision_models import NormalObservation
class BallThrowingSimulator:
    """
    A simple ball simulator that simulates the motion of a ball under gravity.
    And generates observations based on the ball's position.
    The state is represented as [x, y, vx, vy], where (x, y) is the position and (vx, vy) is the velocity.
    The action is gravity, which affects the y-velocity.
    Observations are generated as noisy measurements of the position. 
    The observation noise is assumed to be Gaussian.
    """

    def __init__(self, delta_t=0.5, init_state=None, ball_num=1, observ_model: NormalObservation = None):
        self.delta_t = delta_t
        self.ball_num = ball_num
        self.init_state = init_state if init_state is not None else np.random.rand(
            4, ball_num)*np.array([[20000], [20000], [400], [400]])
        # self.init_state[:, 1] *= np.array([[500], [500], [200], [200]])
        print(self.init_state.shape)

        self.trans_model = BallTransition(delta_t)
        self.observe_model = NormalObservation(
            ball_num) if observ_model is None else observ_model

    def step(self, state: np.ndarray):
        """
        Simulate one step of the ball's motion.
        :param state: The current state of the ball [x, y, vx, vy].
        :return: The next state of the ball after applying the transition model.
        """
        return self.trans_model.propagate(state)
        # return NormalTransition.noisy_propagate(state)

    def observe(self, state: np.ndarray):
        """
        Generate an observation based on the current state of the ball.
        :param state: The current state of the ball [x, y, vx, vy].
        :return: An observation of the ball's position with added Gaussian noise.
        """
        return self.observe_model.observe(state)
    def simulate(self, time=10):
        """
        Simulate the motion of the ball for a given amount of time.
        :param time: The amount of time to simulate.
        :return: A list of states and observations for each step.
        """
        steps = int(time / self.delta_t)
        if steps <= 0:
            raise ValueError("Time must be greater than zero to simulate motion.")
        states = np.zeros((steps+1, 4, self.ball_num))
        states[0, :, :] = self.init_state
        # initial state have no observation
        for i in range(steps):
            states[i+1, :, :] = self.step(states[i:i+1, :, :])

        observations = self.observe_model.observe(states[1:])

        return states, observations
