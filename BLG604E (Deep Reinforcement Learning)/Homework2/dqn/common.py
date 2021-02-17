import gym
import numpy as np

class PrintWriter():
    """ Simple console writer """

    def __init__(self, end_line="\n", flush=False):
        self.end_line = end_line
        self.flush = flush

    def __call__(self, field_dict):
        print(
            ", ".join(key.format(value) for key, value in field_dict.items()),
            end=self.end_line,
            flush=self.flush,
        )


def linear_annealing(init_value, min_value, decay_range):
    """ Decay and return the value at every step linearly.
        Arguments:
            - init_value: Initial value
            - min_value: Minimum value
            - decay_range: Range of the decaying process in terms of
            iterations.
        Return:
            A generator that yields the value and updates it at every step
    """

    result = init_value
    step_dx = (init_value - min_value) / decay_range

    while True:
        result = max(min_value, result - step_dx)
        yield result



def exponential_annealing(init_value, min_value, decay_ratio):
    """ Decay and return the value at every step exponentially.
        Arguments:
            - init_value: Initial value
            - final_value: Minimum value
            - decay_ratio: Decaying rate
        Return:
            A generator that yields the value and updates it at every step
    """
    result = init_value

    while True:
        result = max(min_value, result * decay_ratio)
        yield result


class ResizeAndScalePong(gym.ObservationWrapper):
    """ Observation wrapper that is designed for Pong by Andrej Karpathy.
    Crop, rescale, transpose, and simplify the state.
    """
    def __init__(self, env):
        gym.ObservationWrapper.__init__(self, env)
        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=(1, 80, 80), dtype=np.int8)

    # From Andrew Karpathy
    def observation(self, obs):
        obs = obs[35:195]  # crop
        obs = obs[::2, ::2, 0:1]  # downsample by factor of 2
        obs.transpose(0, 1, 2)
        obs = obs.transpose(2, 0, 1)
        obs[obs == 144] = 0  # erase background (background type 1)
        obs[obs == 109] = 0  # erase background (background type 2)
        obs[obs != 0] = 1  # everything else (paddles, ball) just set to 1
        obs = obs.astype(np.int8)
        return obs


class DerivativeEnv(gym.Wrapper):
    """ Environment wrapper that return the difference between two consecutive
    observations """

    def reset(self, **kwargs):
        self.pre_obs = self.env.reset(**kwargs)
        self.stack_obs = np.array([self.pre_obs, self.pre_obs, self.pre_obs, self.pre_obs])

        return np.zeros(self.observation_space.shape,
                        dtype=self.observation_space.dtype)

    def step(self, ac):
        cur_obs, *remaining = self.env.step(ac)
        self.stack_obs[1:] = self.stack_obs[:3]
        obs = np.concatenate((cur_obs, *self.stack_obs[1:]), axis=0)
        #obs = cur_obs - self.pre_obs
        self.stack_obs[0] = cur_obs
        return (obs, *remaining)


# From Open AI Baseline
class NoopResetEnv(gym.Wrapper):
    def __init__(self, env, noop_max=30, override_num_noops=None):
        """ Sample initial states by taking random number of no-ops on reset.
        No-op is assumed to be action 0.
        """
        gym.Wrapper.__init__(self, env)
        self.noop_max = noop_max
        self.override_num_noops = override_num_noops
        self.noop_action = 0
        assert env.unwrapped.get_action_meanings()[0] == 'NOOP'

    def reset(self, **kwargs):
        """ Do no-op action for a number of steps in [1, noop_max]."""
        self.env.reset(**kwargs)
        if self.override_num_noops is not None:
            noops = self.override_num_noops
        else:
            noops = self.unwrapped.np_random.randint(
                1, self.noop_max + 1)  # pylint: disable=E1101
        assert noops > 0
        obs = None
        for _ in range(noops):
            obs, _, done, _ = self.env.step(self.noop_action)
            if done:
                obs = self.env.reset(**kwargs)
        return obs

    def step(self, ac):
        return self.env.step(ac)


class DoubleActionPong(gym.Wrapper):
    """ Pong specific environment wrapper that reduces the action space into
    two with only meaningful actions (up and down) """

    def __init__(self, env):
        gym.Wrapper.__init__(self, env)
        self.action_space = gym.spaces.Discrete(2)

    def step(self, ac):
        return self.env.step(ac + 2)

    def reset(self, **kwargs):
        return self.env.reset(**kwargs)
