""" Vanilla Replay Buffer
"""
import numpy as np
from collections import namedtuple


class BaseBuffer():
    """ Base class for 1-step buffers. Numpy queue implementation with
    multiple arrays. Sampling efficient in numpy (thanks to fast indexing)

    Arguments:
        - capacity: Maximum size of the buffer
        - state_shape: Shape of a single observation (must be a tuple)
        - state_dtype: Data type of the sumtreestate array. Must be a
        compatible type to numpy
    """

    Transition = namedtuple("Transition",
                            "state action reward next_state terminal")

    def __init__(self, capacity, state_shape, state_dtype):

        self.capacity = capacity

        if not isinstance(state_shape, (tuple, list)):
            raise ValueError("State shape must be a list or a tuple")

        self.transition_info = self.Transition(
            {"shape": state_shape, "dtype": state_dtype},
            {"shape": (1,), "dtype": np.int64},
            {"shape": (1,), "dtype": np.float32},
            {"shape": state_shape, "dtype": state_dtype},
            {"shape": (1,), "dtype": np.float32},
        )

        self.buffer = self.Transition(
            *(np.zeros((capacity, *x["shape"]), dtype=x["dtype"])
              for x in self.transition_info)
        )
    def __len__(self):
        """ Capacity of the buffer
        """
        return self.capacity

    def push(self, transition, *args, **kwargs):
        """ Push a transition object (with single elements) to the buffer
        """
        raise NotImplementedError

    def sample(self, batchsize, *args, **kwargs):
        """ Sample a batch of transitions
        """
        raise NotImplementedError


class UniformBuffer(BaseBuffer):
    """ Standard Replay Buffer that uniformly samples the transitions.
    Arguments:
        - capacity: Maximum size of the buffer
        - state_shape: Shape of a single observation (must be a tuple)
        - state_dtype: Data type of the state array. Must be a compatible
        dtype to numpy
    """

    def __init__(self, capacity, state_shape, state_dtype):
        super().__init__(capacity, state_shape, state_dtype)
        self._cycle = 0
        self.size = 0

    def push(self, transition):
        """ Push a transition object (with single elements) to the buffer.
        FIFO implementation using <_cycle>. <_cycle> keeps track of the next
        available index to write. Remember to update <size> attribute as we
        push transitions.
        """

        action = np.array(transition.action)
        reward = np.array(transition.reward)
        terminal = np.array(transition.terminal*1)
        state = np.array(transition.state)
        next_state = np.array(transition.next_state)
        
        self.buffer.action[self._cycle] = action
        self.buffer.reward[self._cycle] = reward
        self.buffer.terminal[self._cycle] = terminal
        self.buffer.state[self._cycle] = state
        self.buffer.next_state[self._cycle] = next_state

        self._cycle = (self._cycle+1) % self.capacity
        self.size = min(self.size+1, self.capacity)


    def sample(self, batchsize, *args):
        """ Uniformly sample a batch of transitions from the buffer. If
        batchsize is less than the number of valid transitions in the buffer
        return None. The return value must be a Transition object with batch
        of state, actions, .. etc.
            Return: T(states, actions, rewards, terminals, next_states)
        """

        if batchsize > self.size:
            return None

        rand_ids = np.random.choice(range(self.size), batchsize, replace=False)

        ret_state = self.buffer.state[rand_ids]
        ret_action = self.buffer.action[rand_ids]
        ret_reward = self.buffer.reward[rand_ids]
        ret_next_state = self.buffer.next_state[rand_ids]
        ret_terminal = self.buffer.terminal[rand_ids]
                 
        ret_trans = self.Transition(ret_state, ret_action, ret_reward, ret_next_state, ret_terminal)

        return ret_trans

