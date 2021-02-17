import torch
import numpy as np
import gym
import argparse
import sys

from pg.a2c.model import A2C
from pg.a2c.vecenv import ParallelEnv


class GruNet(torch.nn.Module):
    """ Policy and Value network with recurrent (GRU) units.
        Arguments:
            - in_size: Observation size of the environment
            - out_size: Actions space size of the environment
            - hidden: Hidden size of the GRU & other layers (default: 128)
    """

    def __init__(self, in_size, out_size, hidden=128):
        super().__init__()
        self.hidden_size = hidden
        #  /$$$$$$$$ /$$$$$$ /$$       /$$
        # | $$_____/|_  $$_/| $$      | $$
        # | $$        | $$  | $$      | $$
        # | $$$$$     | $$  | $$      | $$
        # | $$__/     | $$  | $$      | $$
        # | $$        | $$  | $$      | $$
        # | $$       /$$$$$$| $$$$$$$$| $$$$$$$$
        # |__/      |______/|________/|________/
        raise NotImplementedError

    def forward(self, state, hx):
        """ Return both the policy and the value outputs as well as hidden
        vector of the recurrent unit.
            Arguments:
                - state: Input tensor
                - hx: Hidden vector of the recurrent unit.
        """
        #  /$$$$$$$$ /$$$$$$ /$$       /$$
        # | $$_____/|_  $$_/| $$      | $$
        # | $$        | $$  | $$      | $$
        # | $$$$$     | $$  | $$      | $$
        # | $$__/     | $$  | $$      | $$
        # | $$        | $$  | $$      | $$
        # | $$       /$$$$$$| $$$$$$$$| $$$$$$$$
        # |__/      |______/|________/|________/
        raise NotImplementedError

        return logits, value, hx


def make_env(envname):
    """ Environment creating function """
    return gym.make(envname)


def main(args):
    """ Start the learning process with the given arguments """
    vecenv = ParallelEnv(args.nenv,
                         lambda: make_env(args.envname))

    # We need to initialize an environment to get dimensions
    env = make_env(args.envname)
    in_size = env.observation_space.shape[0]
    out_size = env.action_space.n

    network = GruNet(in_size, out_size, args.hidden_size)
    optimizer = torch.optim.Adam(network.parameters(), lr=args.lr)
    agent = A2C(network)
    agent.to(args.device)
    # We no longer need to keep this environment
    del env

    agent.learn(args, optimizer, vecenv)


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="A2C with Box2d")
    parser.add_argument("--envname", type=str,
                        default="LunarLander-v2",
                        help="Name of the environment")
    parser.add_argument("--nenv", type=int,
                        help="Number of environemnts run in parallel",
                        default=32)
    parser.add_argument("--lr", type=float, help="Learning rate", default=3e-3)
    parser.add_argument("--device", type=str, help="Torch device",
                        default="cpu")
    parser.add_argument("--n-iteration", type=int,
                        help="Number of iterations to run learning",
                        default=int(1e4))
    parser.add_argument("--n-step", type=int,
                        help="Length of the rollout",
                        default=20)
    parser.add_argument("--hidden-size", type=int,
                        help="Number of neurons in the hidden layers and gru",
                        default=128)
    parser.add_argument("--gamma", type=float,
                        help="Discount factor",
                        default=0.99)
    parser.add_argument("--beta", type=float,
                        help="Entropy coefficient",
                        default=0.3)
    parser.add_argument("--write-period", type=int,
                        help="Logging period",
                        default=100)
    parser.add_argument("--log-window", type=int,
                        help="Last n episodic rewards to log",
                        default=20)
    args = parser.parse_args()

    main(args)
