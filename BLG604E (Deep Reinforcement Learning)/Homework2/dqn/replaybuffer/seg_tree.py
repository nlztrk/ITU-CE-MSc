""" Sum Tree implementation for the prioritized
replay buffer.
"""

import numpy as np


class SegTree():
    """ Base Segment Tree Class with binary heap implementation that push
    values as a Queue(FIFO).
        Arguments:
            - capacity: Maximum size of the tree. It also the number of leaf
            nodes in the tree.
    """

    def __init__(self, capacity):
        self.capacity = capacity
        self._cycle = 0
        self.size = 0

        self.tree = np.zeros(2 * capacity - 1)

    def push(self, value):
        """ Push a value into the tree by calling the update method. Push
        function overrides values when the tree is full  """


        self.update(self._cycle, value)

        self._cycle += 1

        if self._cycle >= self.capacity:
            self._cycle = 0

    def update(self, value):
        raise NotImplementedError


class SumTree(SegTree):
    """ A Binary tree with the property that a parent node is the sum of its
    two children.
        Arguments:
            - capacity: Maximum size of the tree. It also the number of leaf
            nodes in the tree.
    """

    def __init__(self, capacity):
        super().__init__(capacity)

    def get(self, value):
        """ Return the index (ranging from 0 to max capcaity) that corresponds
        to the given value """
        if value > self.tree[0]:
            raise ValueError("Value is greater than the root")

        parent_index = 0

        while True:
            left_child_index = 2 * parent_index + 1
            right_child_index = left_child_index + 1

            # If we reach bottom, end the search
            if left_child_index >= len(self.tree):
                leaf_index = parent_index
                break

            else:  # downward search, always search for a higher priority node
                if value <= self.tree[left_child_index]:
                    parent_index = left_child_index
                else:
                    value -= self.tree[left_child_index]
                    parent_index = right_child_index

        return leaf_index - self.capacity + 1        

    def update(self, index, value):
        """ Update the value of the given index (ranging from 0 to max
        capacity) with the given value
        """
        assert value >= 0, "Value cannot be negative"

        index = index + self.capacity - 1

        change = value - self.tree[index]
        self.tree[index] = value
        
        while index != 0:
            index = (index - 1) // 2  # Round the result to index
            self.tree[index] += change

class MinTree(SegTree):
    """ A Binary tree with the property that a parent node is the minimum of
    its two children.
        Arguments:
            - capacity: Maximum size of the tree. It also the number of leaf
            nodes in the tree.
    """

    def __init__(self, capacity):
        super().__init__(capacity)
        self.tree[:] = np.inf

    def update(self, index, value):
        """ Update the value of the given index (ranging from 0 to max
        capcaity) with the given value
        """
        assert value >= 0, "Value cannot be negative"

        index = index + self.capacity - 1

        self.tree[index] = value

        while index != 0:
            index = (index - 1) // 2  # Round the result to index
            self.tree[index] = min(self.tree[2*index+1], self.tree[2*index+2])


    @property
    def minimum(self):
        """ Return the minimum value of the tree (root node). Complexity: O(1)
        """

        return self.tree[0]
