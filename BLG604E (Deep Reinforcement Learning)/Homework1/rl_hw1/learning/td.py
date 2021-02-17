""" Tabular TD methods
    Author: Anıl Öztürk / 504181504
"""
from collections import defaultdict
from collections import namedtuple
import random
import math
import numpy as np
import time
from collections import deque

from .monte_carlo import TabularAgent


class TabularTDAgent(TabularAgent):
    """ Base class for Tabular TD agents for shared training loop.
    """

    def train(self, env, policy, args):
        """ Training loop for tabular td agents.
        Initiate an episodic reward list. At each episode decrease the epsilon
        value exponentially using args.eps_decay_rate within the boundries of
        args.init_eps and args.final_eps. For every "args._evaluate_period"'th
        step call evaluation function and store the returned episodic reward
        to the list.

        Arguments:
            - env: Warehouse environment
            - policy: Behaviour policy to be used in training(not in
            evaluation)
            - args: namedtuple of hyperparameters

        Return:
            - Episodic reward list of evaluations (not the training rewards)

        **Note**: This function will be used in both Sarsa and Q learning.
        **Note** that: You can also implement you own answer to question 10.
        """

        reward_list = []
        all_eps_reward_list = []

        epsilon = args.init_eps
        np.random.seed(args.seed)
        random.seed(args.seed)
    
        for ix in range(args.episodes):
            
            epsilon = max( min(args.init_eps, epsilon*args.eps_decay_rate), args.final_eps)
            policy = lambda x: self.e_greedy_policy(x, epsilon)
            state = env.reset()
            action = policy(state)

            done = False

            eps_reward = 0.0
            
            while not done:

                next_state, reward, done, _ = env.step(action)
                next_action = policy(next_state)

                transition = (state, action, reward, next_state, next_action)

                self.update(transition, args.alpha, args.gamma)

                eps_reward += reward
                state = next_state
                action = next_action

                if done:
                    break

            all_eps_reward_list.append(eps_reward) 

            if ((ix + 1) % args.evaluate_period) == 0:
                reward_list.append(eps_reward)            
                print("Episode: {}, reward: {}".format(ix + 1, np.mean(all_eps_reward_list[-20:])))

        return reward_list

class QAgent(TabularTDAgent):
    """ Tabular Q leanring agent. Update rule is based on Q learning.
    """

    def __init__(self, nact):
        super().__init__(nact)

    def update(self, transition, alpha, gamma):
        """ Update values of a state-action pair based on the given transition
        and parameters.

        Arguments:
            - transition: 5 tuple of state, action, reward, next_state and
            next_action. "next_action" will not be used in q learning update.
            It is there to be compatible with SARSA update in "train" method.
            - alpha: Exponential decay rate of updates
            - gamma: Discount ratio

        Return:
            temporal diffrence error
        """
        
        state, action, reward, next_state, next_action = transition

        td_error = reward + gamma * self.qvalues[next_state][self.greedy_policy(next_state)] - self.qvalues[state][action]

        self.qvalues[state][action] += alpha * td_error

        return td_error


class SarsaAgent(TabularTDAgent):
    """ Tabular Sarsa agent. Update rule is based on
    SARSA(State Action Reward next_State, next_Action).
    """

    def __init__(self, nact):
        super().__init__(nact)

    def update(self, transition, alpha, gamma):
        """ Update values of a state-action pair based on the given transition
        and parameters.

        Arguments:
            - transition: 5 tuple of state, action, reward, next_state and
            next_action.
            - alpha: Exponential decay rate of updates
            - gamma: Discount ratio

        Return:
            temporal diffrence error
        """

        state, action, reward, next_state, next_action = transition

        td_error = reward + gamma * self.qvalues[next_state][next_action] - self.qvalues[state][action]

        self.qvalues[state][action] += alpha * td_error
