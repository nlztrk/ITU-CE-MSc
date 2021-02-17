""" This module includes DP algorithms.
    Author: Anıl Öztürk / 504181504
"""

class DPAgent():
    r""" Base Dynamic Programming class. DP methods requires the transition
    map in order to optimize policies. This class provides the policy and one
    step policy evaluation as well as policy improvement. It serves as a
    base class for Policy Iteration and Value Iteration algorithms.
    """

    def __init__(self, nact, transitions_map, init_value=0.0):
        self.nact = nact
        self.transitions_map = transitions_map
        self.values = {s: init_value for s in self.transitions_map.keys()}
        self.policy_dist = {s: [1.0/nact] *
                            nact for s in self.transitions_map.keys()}

    def policy(self, state):
        """ Policy pi that returns the action with the highest Q value.
        You can use "policy_dist" that keeps probability values for each
        action at any state. You may have a stochastic policy(random.choice)
        or a determistic one(argmax). Your choice. But for this homework we
        will be using deterministic policy. Therefore, at each state only one
        action will be possible with probability 1.(Initially its stochastic)
        Example policy_dist:
            policy_dist[S] = [1, 0, 0, 0]

        Arguments:
            - state: State/observation of the environment

        Returns:
            action for the given state
        """

        return max(range(self.nact), key = lambda a: self.policy_dist[state][a])

    def one_step_policy_eval(self, gamma=0.95):
        """ One step policy evaluation.
        You can follow the pseudocode given in the "Reinforcement Learing
        Book (Sutton & Barta) chapter 4.1"
        Remember to return delta: Maximum change among all the states

        Arguments:
            - gamma: Discount factor

        Return:
            delta
        """

        delta = 0.

        for key in self.transitions_map.keys():

            ex_val = self.values[key]
            new_val = 0.

            for action in range(self.nact):

                trans_out = self.transitions_map[key][action]
                temp_val = 0.

                for branch in trans_out:
                    temp_val += branch[0] * (branch[2] + gamma * self.values[branch[1]])

                new_val += self.policy_dist[key][action]*temp_val

            self.values[key] = new_val
            
            delta = max(delta, abs(ex_val - self.values[key]))
            
        return delta

    def policy_improvement(self, gamma=0.95):
        """Policy impovement updates the policy according to the most recent
        values. You can follow the ideas given in the "Reinforcement Learing
        (Sutton & Barta) chapter 4.2. Basically, look one step ahead to choose
        the best action.
        Remember to return a boolean value to state if the policy is stable
        Also note that, improved policy must be deterministic. So that, at
        each state only one action is probable(like onehot vector).
        Example policy:
            policy_dist[S] = [0, 1, 0, 0]

        Arguments:
            - gamma: Discount factor

        Return:
            a boolean value stating if stability is reached
        """

        is_policy_stable = True

        for key in self.transitions_map.keys():

            old_action = self.policy(key)
            new_val = [0.] * self.nact

            for action in range(self.nact):

                trans_out = self.transitions_map[key][action]
                temp_val = 0.

                for branch in trans_out:
                    temp_val += branch[0] * (branch[2] + gamma * self.values[branch[1]])

                new_val[action] += temp_val

            max_val = max(new_val)
            maxval_count = new_val.count(max_val)

            for action in range(self.nact):
                self.policy_dist[key][action] = 1 / maxval_count if new_val[action] == max_val else 0

            if self.policy(key) != old_action:
                is_policy_stable = False 

        return is_policy_stable



class PolicyIteration(DPAgent):
    r""" Policy Iteration algorithm that first evaluates the
    values until they converge within epsilon range, then
    updates the policy and repeats the process until the
    policy no longer changes.
    """

    def __init__(self, transitions_map, nact=4):
        super().__init__(nact, transitions_map)

    def optimize(self, gamma, epsilon=0.05, n_iteration=1):
        """ This is the main function where you implement PI.
        Simply use "one_step_policy_eval" and "policy_improvement" methods.
        Itearte as "n_iteration" times in a loop.
        Arguments:
            - gamma: Discount factor
            - epsilon: convergence region of value evaluation
            - n_iteration: Number of iterations
        This should not take more than 10 lines.
        """

        for it in range(n_iteration):

            while self.one_step_policy_eval(gamma) > epsilon:
                continue

            if self.policy_improvement(gamma):
                break

class ValueIteration(DPAgent):
    r""" Value Iteration algorithm iteratively evaluates
    the values and updates the policy until the values
    converges within epsilon range.
    """

    def __init__(self, transitions_map, nact=4):
        super().__init__(nact, transitions_map)

    def optimize(self, gamma, epsilon=0.05, n_iteration=1):
        """ This is the main function where you implement VI.
        Simply use "one_step_policy_eval" and "policy_improvement" methods.
        Itearte as "n_iteration" times in a loop.
        Arguments:
            - gamma: Discount factor
            - epsilon: convergence region of the whole algorithm
        This should not take more than 5 lines.
        """

        while True:

            delta = .0

            for key in self.transitions_map.keys():

                ex_val = self.values[key]
                new_vals = [0.] * self.nact

                for action in range(self.nact):

                    trans_out = self.transitions_map[key][action]
                    temp_val = 0.

                    for branch in trans_out:
                        temp_val += branch[0]*(branch[2]+gamma*self.values[branch[1]])

                    new_vals[action] += temp_val

                max_val = max(new_vals)
                maxval_count = new_vals.count(max_val)
                self.values[key] = max_val

                for action in range(self.nact):
                    self.policy_dist[key][action] = 1 / maxval_count if new_vals[action] == max_val else 0

                delta = max(delta, abs(ex_val - self.values[key]))

            if delta < epsilon:
                break