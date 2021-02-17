import numpy as np
from collections import namedtuple
from itertools import chain

from dqn.replaybuffer.uniform import BaseBuffer
from dqn.replaybuffer.seg_tree import SumTree, MinTree


class PriorityBuffer(BaseBuffer):
    """ Prioritized Replay Buffer that sample transitions with a probability
    that is proportional to their respected priorities.
        Arguments:
            - capacity: Maximum size of the buffer
            - state_shape: Shape of a single observation (must be a tuple)
            - state_dtype: Data type of the state array. Must be a compatible
            dtype to numpy
    """

    def __init__(self, capacity, state_shape, state_dtype,
                 alpha, epsilon=0.1):
        super().__init__(capacity, state_shape, state_dtype)
        self.sumtree = SumTree(capacity)
        self.mintree = MinTree(capacity)
        self.epsilon = epsilon
        self.alpha = alpha
        self._cycle = 0
        self.size = 0

        self.max_p = epsilon ** alpha

        self.priorities = np.zeros((capacity, ), dtype=np.float32)


    def push(self, transition):
        """ Push a transition object (with single elements) to the buffer.
        Transitions are pushed with the current maximum priority (also push
        priorities to both min and sum tree). Remember to set <_cycle> and
        <size> attributes.
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


        self.sumtree.push(self.max_p)
        self.mintree.push(self.max_p)
        self.priorities[self._cycle] = self.max_p

        self._cycle = (self._cycle+1) % self.capacity
        self.size = min(self.size+1, self.capacity)

        #self.max_p = np.max(self.sumtree.tree[-self.sumtree.capacity:])

        #  /$$$$$$$$ /$$$$$$ /$$       /$$
        # | $$_____/|_  $$_/| $$      | $$
        # | $$        | $$  | $$      | $$
        # | $$$$$     | $$  | $$      | $$
        # | $$__/     | $$  | $$      | $$
        # | $$        | $$  | $$      | $$
        # | $$       /$$$$$$| $$$$$$$$| $$$$$$$$
        # |__/      |______/|________/|________/
        #raise NotImplementedError

    def sample(self, batch_size, beta):
        """ Sample a transition based on priorities.
            Arguments:
                - batch_size: Size of the batch
                - beta: Importance sampling weighting annealing
            Return:
                - batch of samples
                - indexes of the sampled transitions (so that corresponding
                priorities can be updated)
                - Importance sampling weights
        """
        if batch_size > self.size:
            return None

        if self.size == self.capacity:
            prios = self.priorities
        else:
            prios = self.priorities[:self._cycle]

        probs = np.array(prios, dtype=np.float32) ** self.alpha

        probs /= probs.sum()
        
        rand_ids = np.random.choice(self.size, batch_size, p=probs, replace=False)
        rand_ids = np.sort(rand_ids)

        ret_state = self.buffer.state[rand_ids]
        ret_action = self.buffer.action[rand_ids]
        ret_reward = self.buffer.reward[rand_ids]
        ret_next_state = self.buffer.next_state[rand_ids]
        ret_terminal = self.buffer.terminal[rand_ids]       
        ret_trans = self.Transition(ret_state, ret_action, ret_reward, ret_next_state, ret_terminal)

        weights = (self.size * probs[rand_ids]) ** (-beta)
        weights /= weights.max()
        
        return ret_trans, rand_ids, weights

        #  /$$$$$$$$ /$$$$$$ /$$       /$$
        # | $$_____/|_  $$_/| $$      | $$
        # | $$        | $$  | $$      | $$
        # | $$$$$     | $$  | $$      | $$
        # | $$__/     | $$  | $$      | $$
        # | $$        | $$  | $$      | $$
        # | $$       /$$$$$$| $$$$$$$$| $$$$$$$$
        # |__/      |______/|________/|________/
        #raise NotImplementedError

    def update_priority(self, indexes, values):
        """ Update the priority values of given indexes (for both min and sum
        trees). Remember to update max_p value! """


        #values = np.array([value+0.001 for value in values])
        #values += self.epsilon
        #priorities = np.power(values, self.max_p)

        #self.max_p = max(values)


        # ids = np.argsort(indexes)[::-1]
        # indexes = np.array(indexes)[ids]
        # values = np.array(values)[ids]

         

        for index, value in zip(indexes, values):
            self.priorities[index] = value
            value  = (value + self.epsilon) ** self.alpha
            self.mintree.update(index, value)
            self.sumtree.update(index, value)

        self.max_p = np.max(self.sumtree.tree[-self.sumtree.capacity:])
