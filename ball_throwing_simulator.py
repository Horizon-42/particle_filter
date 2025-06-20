# simulate motion and observation
import numpy as np
from transition_models import NormalTransition
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
    def __init__(self, delta_t=0.5, init_state=None, ball_num=1, observation_variance=[10, 10]):
        self.delta_t = delta_t
        self.ball_num = ball_num
        self.init_state = init_state if init_state is not None else np.random.rand(
            4, ball_num)*np.array([[50], [50], [200], [200]])
        print(self.init_state.shape)

    def step(self, state):
        """
        Simulate one step of the ball's motion.
        :param state: The current state of the ball [x, y, vx, vy].
        :return: The next state of the ball after applying the transition model.
        """
        return NormalTransition.propagate(state)
        # return NormalTransition.noisy_propagate(state)

    def observe(self, state: np.ndarray):
        """
        Generate an observation based on the current state of the ball.
        :param state: The current state of the ball [x, y, vx, vy].
        :return: An observation of the ball's position with added Gaussian noise.
        """
        return NormalObservation.observe(state)
    def simulate(self, time=10):
        """
        Simulate the motion of the ball for a given amount of time.
        :param time: The amount of time to simulate.
        :return: A list of states and observations for each step.
        """
        states = []
        observations = []
        steps = int(time / self.delta_t)
        if steps <= 0:
            raise ValueError("Time must be greater than zero to simulate motion.")
        state = self.init_state.copy()
        states.append(state)
        observations.append(self.observe(state))
        for _ in range(steps):
            state = self.step(state)
            # if state[1]< 0: # touch the ground
            #     break
            states.append(state)
            observations.append(self.observe(state))

        return np.array(states), np.array(observations)