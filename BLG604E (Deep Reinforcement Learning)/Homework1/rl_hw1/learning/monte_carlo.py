""" Tabular MC algorithms
    Author: Anıl Öztürk / 504181504
"""
from collections import defaultdict
import random
import math
from collections import deque
import numpy as np
import time


class TabularAgent:

    r""" Base class for the tabular agents. This class
    provides policies for the tabular agents.
    """

    def __init__(self, nact):
        self.qvalues = defaultdict(lambda: [0.0]*nact)
        self.nact = nact
        random.seed()
    
    def greedy_policy(self, state, *args, **kwargs):
        # Returns the best action.
        return max(range(self.nact), key = lambda a: self.qvalues[state][a])

    def e_greedy_policy(self, state, epsilon, *args, **kwargs):
        # Returns the best action with the probability (1 - e) and 
        # action with probability e
        return self.greedy_policy(state) if random.random() > epsilon else random.randrange(self.nact)

    def evaluate(self, env, render=False):
        """ Single episode evaluation of the greedy agent.
        Arguments:
            - env: Warehouse or Mazeworld environemnt
            - render: If true render the environment(default False)
        Return:
            Episodic reward
        """

        state = env.reset()
        done = False

        episodic_reward = 0.

        while not done:

            action = self.greedy_policy(state)
            state, reward, done, info = env.step(action)

            if not hasattr(self, 'eval_for_mc_sim'):
                time.sleep(1/10)

            if render:
                env.render()

            episodic_reward += reward

        return episodic_reward


class MonteCarloAgent(TabularAgent):
    """ Tabular Monte Carlo Agent that updates q values based on MC method.
    """

    def __init__(self, nact):
        super().__init__(nact)

    def one_epsiode_train(self, env, policy, gamma, alpha):
        """ Single episode training function.
        Arguments:
            - env: Mazeworld environment
            - policy: Behaviour policy for the training loop
            - gamma: Discount factor
            - alpha: Exponential decay rate of updates

        Returns:
            episodic reward

        **Note** that in the book (Sutton & Barto), they directly assign the
        return to q value. You can either implmenet that algorithm (given in
        chapter 5) or use exponential decaying update (using alpha).
        """

        self.gamma = gamma
        self.alpha = alpha
        

        if not hasattr(self, 'rewards'):
            self.rewards = defaultdict(lambda: [0.]*self.nact)
            self.rewards_count = defaultdict(lambda: [0.]*self.nact)

        episode = []
        state = env.reset()
        done = False

        while not done:

            action = policy(state)
            next_state, reward, done, _ = env.step(action)
            episode.append((state, action, reward))
            
            if done:
                break

            state = next_state

        states_in_episode = [(x[0], x[1]) for x in episode]

        for state, action in states_in_episode:

            first_occurence_idx = next(i for i,x in enumerate(episode) if x[0] == state)

            G = sum([x[2]*(self.gamma**i) for i,x in enumerate(episode[first_occurence_idx:])])
            
            self.rewards[state][action] += G
            self.rewards_count[state][action] += 1.
            self.qvalues[state][action] = (1-self.alpha) * self.qvalues[state][action] + self.alpha *(self.rewards[state][action] / self.rewards_count[state][action])

        self.eval_for_mc_sim = "defined"
        ret =  self.evaluate(env, render=False)
        del self.eval_for_mc_sim

        return ret
