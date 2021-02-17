import torch
import numpy as np
import gym
import argparse
from collections import namedtuple

from dqn.rainbow.model import RainBow
from dqn.rainbow.train import Trainer
from dqn.rainbow.layers import HeadLayer


class ValueNet(torch.nn.Module):

    def __init__(self, in_size, out_size, extensions):
        super().__init__()
        """ Fully connected neural network to estimate values with enhanced
        head layer.
        Arguments:
            - in_size: Size of a state
            - out_size: Action size
            - extensions: A dictionary that keeps extension arguments
        """
        if extensions["distributional"]:
            self.natoms = extensions["distributional"]["natoms"]

        self.out_size = out_size
        self.extensions = extensions

        self.feature = torch.nn.Sequential(
            torch.nn.Linear(in_size, 128),
            torch.nn.ReLU(),
            torch.nn.Linear(128, 128),
            torch.nn.ReLU(),

        )

        self.head = HeadLayer(128, out_size, extensions, 64)

    def forward(self, state):
        features = self.feature(state)
        out = self.head(features)
        
        if self.extensions["distributional"]:
            out = torch.nn.functional.softmax(out.view(-1, self.natoms), 1).view(-1, self.out_size, self.natoms)
        
        return out


def make_env(envname):
    """ Environment creating function """
    env = gym.make(args.envname)
    return env


def main(args):
    """ The main function that prepares a model, extension dictionary and
    starts training """
    env = make_env(args.envname)
    env._max_episode_steps = args.max_episode_len
    state_shape = env.observation_space.shape
    state_dtype = env.observation_space.dtype
    act_size = env.action_space.n

    extensions = {
        "double": not args.no_double,
        "dueling": not args.no_dueling,
        "noisy": False if args.no_noisy else {
            "init_std": args.noisy_std
        },
        "nstep": args.n_steps,
        "distributional": False if args.no_dist else {
            "vmin": args.vmin,
            "vmax": args.vmax,
            "natoms": args.natoms,
        },
        "prioritized": False if args.no_prioritized else {
            "alpha": args.alpha,
            "beta_init": args.beta_init
        },
    }

    valuenet = ValueNet(
        state_shape[0],
        act_size,
        extensions
    ).to(args.device)

    agent = RainBow(valuenet, act_size, extensions,
                    args.buffer_capacity, state_shape, state_dtype)
    optimizer = torch.optim.Adam(valuenet.parameters(), lr=args.lr)
    agent.to(args.device)
    Trainer(args, agent, optimizer, env)()


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Rainbow')
    parser.add_argument("--envname", type=str,
                        default="LunarLander-v2",
                        help="Name of the environment")
    parser.add_argument("--n-iterations", type=int, default=300000,
                        help="Number of training iterations")
    parser.add_argument("--start-update", type=int, default=100,
                        help="Number of iterations until starting to update")
    parser.add_argument("--max-episode-len", type=int, default=1000,
                        help="Maximum length of an episode before termination")

    # ----------------------- Hyperparameters -----------------------
    parser.add_argument("--batch-size", type=int, default=32,
                        help="Batch size of each update in training")
    parser.add_argument("--gamma", type=float, default=0.99,
                        help="Discount Factor")
    parser.add_argument("--lr", type=float, default=1e-3,
                        help="Learning rate")
    parser.add_argument("--device", type=str, default="cuda",
                        help="Torch device")
    parser.add_argument("--target-update-period", type=int, default=500,
                        help="Target network updating period")
    parser.add_argument("--buffer-capacity", type=int, default=50000,
                        help="Replay buffer capacity")
    parser.add_argument("--epsilon-init", type=float, default=1,
                        help="Initial value of the epsilon")
    parser.add_argument("--epsilon-min", type=float, default=0.01,
                        help="Minimum value of the epsilon")
    parser.add_argument("--epsilon-decay", type=float, default=0.925,
                        help="Epsilon decay rate for exponential decaying")
    parser.add_argument("--epsilon-range", type=int, default=None,
                        help="Epsilon decaying range for linear decay")
    parser.add_argument("--clip-grad", action="store_true", default=False,
                        help="Gradient Clip between -1 and 1. Default: No")

    # ----------------------- Extensions -----------------------
    parser.add_argument("--no-double", action="store_true",
                        help="Disable double DQN extension")
    parser.add_argument("--no-dueling", action="store_true",
                        help="Disable dueling DQN extension")

    parser.add_argument("--no-noisy", action="store_true",
                        help="Disable noisy layers")
    parser.add_argument("--noisy-std", type=float, default=0.5,
                        help="Initial std for noisy layers")

    parser.add_argument("--no-prioritized", action="store_true",
                        help="Disable prioritized replay buffer")
    parser.add_argument("--alpha", type=float, default=0.6,
                        help="Prioritization exponent")
    parser.add_argument("--beta-init", type=float, default=0.4,
                        help="Prioritization exponent")

    parser.add_argument("--n-steps", type=int, default=1,
                        help="Number of steps for bootstrapping")

    parser.add_argument("--no-dist", action="store_true",
                        help="Disable distributional DQN extension")
    parser.add_argument("--vmin", type=float, default=-100,
                        help="Minimum value for distributional DQN extension")
    parser.add_argument("--vmax", type=float, default=100,
                        help="Maximum value for distributional DQN extension")
    parser.add_argument("--natoms", type=int, default=51,
                        help="Number of atoms in distributional DQN extension")

    # ----------------------- Miscelenious -----------------------
    parser.add_argument("--eval-period", type=int, default=1000,
                        help="Evaluation period in terms of iteration")
    parser.add_argument("--eval-episode", type=int, default=3,
                        help="Number of episodes to evaluate")
    parser.add_argument("--save-model", action="store_true",
                        help="If given most successful models so far will be saved")
    parser.add_argument("--eval-mode", action="store_true", default=False,
                        help="Demonstration mode")
    parser.add_argument("--model-dir", type=str, default="models/",
                        help="Directory to save models")
    parser.add_argument("--write-period", type=int, default=100,
                        help="Writer period")

    args = parser.parse_args()

    main(args)
