import unittest
import numpy as np

from dqn.replaybuffer.uniform import UniformBuffer
from dqn.replaybuffer.prioritized import PriorityBuffer
from dqn.replaybuffer.seg_tree import SumTree, MinTree


class TestVanillaBuffer(unittest.TestCase):

    def setUp(self):
        self.trans = UniformBuffer.Transition(
            np.ones(24, dtype="float"),
            1,
            1.0,
            -np.ones(24, dtype="float"),
            True,
        )

    def test_sample(self):
        capacity = 1000
        buffer = UniformBuffer(capacity, (24,), "float")
        for i in range(10):
            buffer.push(self.trans)
        sample = buffer.sample(5)
        self.assertIsInstance(sample, UniformBuffer.Transition)

        self.assertEqual(buffer.sample(11), None)

        for item, info in zip(sample, buffer.transition_info):
            self.assertEqual(item.shape, (5, *info["shape"]))
            self.assertEqual(item.dtype, info["dtype"])

        for i in range(100):
            self.assertEqual(buffer.sample(6).state.sum(), 24.0 * 6)

    def test_cycle_nd_size(self):
        capacity = 1000
        buffer = UniformBuffer(capacity, (24,), "float")
        for i in range(capacity * 2):
            self.assertEqual(i % capacity, buffer._cycle)
            buffer.push(self.trans)
            self.assertEqual(min(i+1, capacity), buffer.size)
            self.assertEqual(buffer.buffer.state.sum().item() // 24,
                             min(i+1, capacity))


class TestSumTree(unittest.TestCase):

    def setUp(self):
        pass

    def test_init(self):
        """ Test the sum tree size (not the capacity)"""
        for s, value in zip((-1, 0, 1), (2045, 2047, 2049)):
            tree = SumTree(2**10 + s)
            self.assertEqual(tree.tree.shape[0], value)

    def test_update(self):
        """ Test update within a sum tree with capacity 15"""
        tree = SumTree(15)
        tree.update(7, 10)
        
        self.assertEqual(np.all(
            tree.tree == [10, 10,  0,  0, 10,  0,  0,  0,  0,  0,
                          10,  0,  0,  0,  0,  0,  0,  0, 0,  0,
                          0, 10,  0,  0,  0,  0,  0,  0,  0, ]), 1)

    def test_push(self):
        """ Test push and tree values """
        tree = SumTree(8)
        for i in range(8):
            tree.push(i+1)
        self.assertEqual(np.all(
            tree.tree == [36, 10, 26,  3,  7, 11, 15,  1,  2,  3,
                          4,  5,  6,  7,  8, ]), 1)

        """ Test if push is a FIFO """
        tree = SumTree(7)
        for i in range(10):
            tree.push(i+1)
        self.assertEqual(np.all(
            tree.tree == [49, 28, 21, 19,  9, 13,  8,  9, 10,  4,
                          5,  6,  7, ]), 1)

    def test_push_nd_get(self):
        """ Test sum tree behaviour """
        from collections import defaultdict
        tree = SumTree(80)
        for i in range(80):
            tree.push(i % 2)

        freq = defaultdict(int)
        for j in range(80 * 1000):
            ix = tree.get(np.random.uniform(0, tree.tree[0]))
            freq[ix] += 1

        for j in range(80):
            if j % 2 == 0:
                self.assertEqual(freq[j], 0)
            else:
                self.assertAlmostEqual(freq[j], 2000, delta=300)

    def get_nd_update(self):
        """ Test update """
        tree = SumTree(9)
        tree.update(4, 1)
        self.assertEqual(tree.get(0.5), 4)
        tree.update(4, 0)
        tree.update(5, 1)
        self.assertEqual(tree.get(0.5), 5)


class TestMinTree(unittest.TestCase):

    def setUp(self):
        pass

    def test_push(self):
        """ Test push and tree values """
        tree = MinTree(8)
        for i in range(8):
            tree.push(i+1)
        self.assertEqual(np.all(
            tree.tree == [1, 1, 5, 1, 3, 5, 7, 1, 2, 3,
                          4, 5, 6, 7, 8, ]), 1)

    def test_update(self):
        """ Test update method"""
        tree = MinTree(7)
        for i in range(6):
            tree.push(i+2)
        self.assertEqual(tree.minimum, 2)
        tree.update(0, 1)
        self.assertEqual(tree.minimum, 1)
        tree.update(0, 5)
        self.assertEqual(tree.minimum, 3)


class TestPrioritizedBuffer(unittest.TestCase):

    def setUp(self):
        self.trans_pos = UniformBuffer.Transition(
            np.ones(24, dtype="float"),
            1,
            1.0,
            -np.ones(24, dtype="float"),
            True,
        )

        self.trans_neg = UniformBuffer.Transition(
            -np.ones(24, dtype="float"),
            1,
            1.0,
            np.ones(24, dtype="float"),
            True,
        )

    def test_overwriting(self):
        """ Test if Priority Buffer is overwriting """
        buffer = PriorityBuffer(500, (24,), "float", alpha=0.5, epsilon=4)
        for i in range(500):
            buffer.push(self.trans_pos)
        for i in range(500):
            buffer.push(self.trans_neg)

        self.assertEqual(buffer.sumtree.tree[0], 1000)
        batch, indexes, weights = buffer.sample(500, 0.5)
        self.assertEqual(batch.state.sum() / 24, -500)

    def test_sample_indexes(self):
        """ Test if indexes are overflowing """
        buffer = PriorityBuffer(500, (24,), "float", alpha=0.5, epsilon=4)
        for i in range(100):
            buffer.push(self.trans_pos)
        for k in range(10):
            batch, indexes, weights = buffer.sample(10, 0.5)
            self.assertEqual(np.all(
                indexes < 100
            ), 1)

    def test_update(self):
        """ Test update_priority method and max_priority """
        buffer = PriorityBuffer(7, (24,), "float", alpha=0.5, epsilon=4)
        for i in range(4):
            buffer.push(self.trans_pos)
        
        buffer.update_priority(
            [0, 1], [12, 12]
        )
        
        self.assertEqual(buffer.sumtree.tree[0], 12)
        self.assertEqual(buffer.mintree.minimum, 2)
        buffer.push(self.trans_pos)
        self.assertEqual(buffer.sumtree.tree[0], 16)

    def test_weights(self):
        """ Test weights returned by sample method """
        buffer = PriorityBuffer(9, (24,), "float", alpha=0.5, epsilon=4)
        for i in range(4):
            buffer.push(self.trans_pos)

        batch, indexes, weights = buffer.sample(4, 0.5)
        self.assertEqual(np.all(
            weights == np.array([1, 1, 1, 1])
        ), 1)

        # Segmented sampling guarantees following order:
        self.assertEqual(np.all(
            indexes == np.arange(4)
        ), 1)


if __name__ == '__main__':
    unittest.main()
