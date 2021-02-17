""" Function apporximation methods in TD
    Author: Anıl Öztürk / 504181504
"""
import numpy as np
import time
from collections import namedtuple
import random


class ApproximateAgent():
    r""" Base class for the approximate methods. This class
    provides policies and training loop. Initiate a weight matrix
    with shape (#observation x #action).
    """

    def __init__(self, nobs, nact):
        self.nact = nact
        self.nobs = nobs
        self.weights = np.random.uniform(-0.1, 0.1, size=(nobs, nact)).astype(np.float64)

    def q_values(self, state):
        """ Return the q values of the given state for each action.
        """
        return np.dot(state, self.weights)

    def greedy_policy(self, state, *args):
        """ Return the best possible action according to the value
        function """

        return np.argmax(self.q_values(state))

    def e_greedy_policy(self, state, epsilon):
        """ Policy that returns the best action according to q values with
        (epsilon/#action) + (1 - epsilon) probability and any other action with
        probability episolon/#action.
        """

        return self.greedy_policy(state) if random.random() > epsilon else random.randrange(self.nact)
        

    def train(self, env, policy, args):
        """ Training loop for the approximate agents.
        Initiate an episodic reward list and a loss list. At each episode
        decrease the epsilon value exponentially using args.eps_decay_rate
        within the boundries of the args.init_eps and args.final_eps. At each
        transition update the agent(weights of the function). For every
        "args._evaluate_period"'th step call evaluation function and store the
        returned episodic reward to the reward list.

        Arguments:
            - env: gym environment
            - policy: Behaviour policy to be used in training(not in
            evaluation)
            - args: namedtuple of hyperparameters

        Return:
            - Episodic reward list of evaluations (not the training rewards)
            - Loss list of the training (one loss for per update)
        """

        reward_list = []
        all_eps_reward_list = []
        all_its_loss_list = []

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

                if done:
                    next_action = False

                transition = (state, action, reward, next_state, next_action)

                loss = self.update(transition, args.alpha, args.gamma)
                all_its_loss_list.append(loss)

                eps_reward += reward
                state = next_state
                action = next_action

                if done:
                    break

            all_eps_reward_list.append(eps_reward) 

            if ((ix + 1) % args.evaluate_period) == 0:
                reward_list.append(eps_reward)            
                print("Episode: {}, reward: {}".format(ix + 1, np.mean(all_eps_reward_list[-20:])))

        return reward_list, all_its_loss_list

    def update(self, *arg, **kwargs):
        raise NotImplementedError

    def evaluate(self, env):
        raise NotImplementedError


class ApproximateQAgent(ApproximateAgent):
    r""" Approximate Q learning agent where the learning is done
    via minimizing the mean squared value error with semi-gradient descent.
    This is an off-policy algorithm.
    """

    def __init__(self, nobs, nact):
        super().__init__(nobs, nact)

    def update(self, transition, alpha, gamma):
        """ Update the parameters that parameterized the value function
        according to (semi-gradient) q learning.

        Arguments:
            - transition: 4 tuple of state, action, reward and next_state
            - alpha: Learning rate of the update function
            - gamma: Discount rate

        Return:
            Mean squared temporal difference error
        """

        state, action, reward, next_state, next_action = transition

        reward = np.float64(reward)

        grad = np.array(np.gradient(self.weights)[0],dtype=np.float64)
        mean_squared_td_error = np.float64(.0)

        if not next_action:
            mean_squared_td_error = (reward - self.q_values(state)[action]) * grad
        else:
            mean_squared_td_error = (reward + gamma * self.q_values(next_state)[next_action] - self.q_values(state)[action]) * grad
        
        self.weights += alpha * mean_squared_td_error

        return mean_squared_td_error

    def evaluate(self, env, render=False):
        """ Single episode evaluation of the greedy agent.
        Arguments:
            - env: gym environemnt
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

            if render:

                if not hasattr(self, 'eval_for_mc_sim'):
                    time.sleep(1/10)
                    
                env.render(mode='close')

            episodic_reward += reward

        env.close()

        return episodic_reward