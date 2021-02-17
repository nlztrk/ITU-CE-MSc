import torch
import numpy as np
from collections import namedtuple

from pg.common import PrintWriter


class Reinforce(torch.nn.Module):
    """ Simple Policy Gradient algorithm that does not rely on value learning.
        Arguments:
            - policynet: A neural network that represents the policy and
            returns a categorical distribution when called with a state.
    """

    Transition = namedtuple("Transition", "logprob reward")

    def __init__(self, policynet):
        super().__init__()
        self.policynet = policynet
        self.writer = PrintWriter(flush=True)

    def forward(self, state):
        """ Get the Categrical distribution that denote the policy pi(s) at
        state s. """
        return self.policynet(state)

    def accumulate_gradient(self, rollout, gamma):
        """ Calculate gradient by performing backprob over the entire rollout.
        Every transition in the rollout must be used in the gradient
        calculation.
            Arguments:
                - rollout: A list of Transitions
                - gamma: Discount factor
        """
        R = 0
        for logprob, reward in rollout[::-1]:
            #  /$$$$$$$$ /$$$$$$ /$$       /$$
            # | $$_____/|_  $$_/| $$      | $$
            # | $$        | $$  | $$      | $$
            # | $$$$$     | $$  | $$      | $$
            # | $$__/     | $$  | $$      | $$
            # | $$        | $$  | $$      | $$
            # | $$       /$$$$$$| $$$$$$$$| $$$$$$$$
            # |__/      |______/|________/|________/
            raise NotImplementedError

    def learn(self, args, opt, env):
        """ The learner function similar is to sklearn API. In a loop, create a
        rollout that consists of all the transitions of an episode. Use this
        rollout to accumulate gradients. Then, update the policy network by
        stepping the optimizer.
            Arguments:
                - args: Parsed command line arguments
                - opt: The optimizer to update the policy parameters
                - env: Enviornment object
        """
        episodic_rewards = []

        for ix in range(args.n_episodes):
            episode_reward = 0
            state = env.reset()
            rollout = []
            for jx in range(args.max_episode_len):

                #  /$$$$$$$$ /$$$$$$ /$$       /$$
                # | $$_____/|_  $$_/| $$      | $$
                # | $$        | $$  | $$      | $$
                # | $$$$$     | $$  | $$      | $$
                # | $$__/     | $$  | $$      | $$
                # | $$        | $$  | $$      | $$
                # | $$       /$$$$$$| $$$$$$$$| $$$$$$$$
                # |__/      |______/|________/|________/
                raise NotImplementedError

            opt.zero_grad()
            self.accumulate_gradient(rollout, args.gamma)
            opt.step()

            episodic_rewards.append(episode_reward)

            if (ix + 1) % args.write_period == 0:
                self.writer(
                    [
                        self.writer.Column("Episode", "{:7}", ix + 1),
                        self.writer.Column("Reward", "{:7.3f}", np.mean(
                            episodic_rewards[-args.log_window:])),
                    ],
                )
