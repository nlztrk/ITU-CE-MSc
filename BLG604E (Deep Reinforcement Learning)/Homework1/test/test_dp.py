import unittest
import numpy as np

from rl_hw1.dp import make_transition_map, DPAgent


class TestDPMethods(unittest.TestCase):

    board = np.array([
        [35, 35, 35, 35, 35, 35, 35],
        [35, 32, 32, 32, 32, 64, 35],
        [35, 32, 35, 32, 32, 32, 35],
        [35, 32, 32, 35, 32, 32, 35],
        [35, 32, 32, 32, 32, 32, 35],
        [35, 80, 32, 32, 32, 32, 35],
        [35, 35, 35, 35, 35, 35, 35]
    ])

    def test_make_transition_map(self):

        tmap = make_transition_map(self.board)

        assert tmap[(1, 1)][1] == [(0.1, (1, 1), 0.0, False),
                                   (0.7, (2, 1), 0.0, False),
                                   (0.1, (1, 1), 0.0, False),
                                   (0.1, (1, 2), 0.0, False)]

        assert tmap[(1, 4)][3] == [(0.1, (1, 4), 0.0, False),
                                   (0.1, (2, 4), 0.0, False),
                                   (0.1, (1, 3), 0.0, False),
                                   (0.7, (1, 5), 1.0, True)]

        assert tmap[(2, 5)][0] == [(0.7, (1, 5), 1.0, True),
                                   (0.1, (3, 5), 0.0, False),
                                   (0.1, (2, 4), 0.0, False),
                                   (0.1, (2, 5), 0.0, False)]

    def test_policy_eval(self):

        tmap = make_transition_map(self.board)
        agent = DPAgent(4, tmap)

        agent.one_step_policy_eval(gamma=1)
        self.assertEqual(agent.values[(1, 4)], 0.25)
        agent.one_step_policy_eval(gamma=1)
        self.assertEqual(agent.values[(2, 5)], 0.3125)

    def test_policy_imporvement(self):
        
        tmap = make_transition_map(self.board)
        agent = DPAgent(4, tmap)
        agent.one_step_policy_eval()
        agent.policy_improvement()
        assert agent.policy_dist[(2, 5)] == [1, 0, 0, 0]
        assert agent.policy_dist[(1, 4)] == [0, 0, 0, 1]
        agent.one_step_policy_eval()
        agent.policy_improvement()
        assert agent.policy_dist[(1, 2)] == [0, 0, 0, 1]


if __name__ == '__main__':
    unittest.main()
