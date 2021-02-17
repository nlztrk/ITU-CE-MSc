import gym
import torch
import numpy as np
import argparse

from model import Reinforce


class PolicyNet(torch.nn.Module):
    """ Simple policy network that returns a Categorical distribution.
        Arguments:
            - in_size: Observation size of the environment
            - act_size: Actions space size of the environment
            - hidden_size: Hidden size of the layers (default: 128)
    """

    def __init__(self, in_size, act_size, hidden_size=128):
        super().__init__()
        #  /$$$$$$$$ /$$$$$$ /$$       /$$
        # | $$_____/|_  $$_/| $$      | $$
        # | $$        | $$  | $$      | $$
        # | $$$$$     | $$  | $$      | $$
        # | $$__/     | $$  | $$      | $$
        # | $$        | $$  | $$      | $$
        # | $$       /$$$$$$| $$$$$$$$| $$$$$$$$
        # |__/      |______/|________/|________/
        raise NotImplementedError

    def forward(self, state):

        #  /$$$$$$$$ /$$$$$$ /$$       /$$
        # | $$_____/|_  $$_/| $$      | $$
        # | $$        | $$  | $$      | $$
        # | $$$$$     | $$  | $$      | $$
        # | $$__/     | $$  | $$      | $$
        # | $$        | $$  | $$      | $$
        # | $$       /$$$$$$| $$$$$$$$| $$$$$$$$
        # |__/      |______/|________/|________/
        raise NotImplementedError

        return dist  # You can use torch.distributions.categorical.Categorical


def main(args):
    """ Start the learning process with the given arguments """
    env = gym.make(args.envname)

    in_size = env.observation_space.shape[0]
    act_size = env.action_space.n

    policynet = PolicyNet(in_size,
                          act_size,
                          hidden_size=args.hidden_size)
    agent = Reinforce(policynet)
    opt = torch.optim.Adam(agent.parameters(), lr=args.lr)

    agent.to(args.device)
    agent.learn(args, opt, env)


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description="REINFOCE with Box2d")
    parser.add_argument("--envname", type=str,
                        default="CartPole-v1",
                        help="Name of the environment")
    parser.add_argument("--n-episodes", type=int, default=1000,
                        help="Number of training episodes")
    parser.add_argument("--max-episode-len", type=int, default=1000,
                        help="Maximum length of an episode before termination")
    parser.add_argument("--write-period", type=int, default=20,
                        help="Maximum length of an episode before termination")

    parser.add_argument("--gamma", type=float, default=0.99,
                        help="Discount Factor")
    parser.add_argument("--lr", type=float, default=0.001,
                        help="Discount Factor")
    parser.add_argument("--hidden-size", type=int, default=128,
                        help=("Number of neurons in the hidden layers of the "
                              "policy network"))
    parser.add_argument("--device", type=str, default="cpu",
                        help="Torch device")
    parser.add_argument("--clip-grad", action="store_true",
                        help="Gradient Clip between -1 and 1. Default: No")
    parser.add_argument("--log-window", type=int,
                        help="Last n episodic rewards to log",
                        default=20)

    args = parser.parse_args()

    main(args)
