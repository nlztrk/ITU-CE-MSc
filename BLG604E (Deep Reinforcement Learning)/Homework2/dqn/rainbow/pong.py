import torch
import numpy as np
import gym
import argparse
from collections import namedtuple

from dqn.rainbow.model import RainBow
from dqn.rainbow.train import Trainer
from dqn.rainbow.layers import HeadLayer
from dqn.common import ResizeAndScalePong, NoopResetEnv, DerivativeEnv, DoubleActionPong


class ValueNet(torch.nn.Module):
    """ Convolutional neural network with a Head layer to estimate
    values. The spatial size is expected to be (80 x 80).
        Arguments:
            - in_size: Channel size of the input
            - out_size: Action size
            - extensions: A dictionary that keeps extension arguments
    """
    def __init__(self, in_size, out_size, extensions):
        super().__init__()

        if extensions["distributional"]:
            self.natoms = extensions["distributional"]["natoms"]

        self.out_size = out_size
        self.extensions = extensions

        self.feature = torch.nn.Sequential(
            torch.nn.Conv2d(in_size, 32, kernel_size=8, stride=4),
            torch.nn.ReLU(),
            torch.nn.Conv2d(32, 64, kernel_size=4, stride=2),
            torch.nn.ReLU(),
            torch.nn.Conv2d(64, 64, kernel_size=3, stride=1),
            torch.nn.ReLU(),

        )

        self.linear = torch.nn.Sequential(
            torch.nn.Linear(64 * 6 * 6, 512),
            torch.nn.ReLU()
        )

        self.head = HeadLayer(512, out_size, extensions, 64)

    def forward(self, state):

        if len(state.shape)==3:
            ch_size = state.shape[0]
            state_h = state.shape[1]
            state_w = state.shape[2]
            state = state.view(1, ch_size, state_h, state_w)       

        assert tuple(state.shape[-2:]) == (80, 80), state.shape

        features = self.feature(state)
        
        features = features.reshape(features.shape[0], -1)
        features = self.linear(features)
        out = self.head(features)
        
        if self.extensions["distributional"]:
            out = torch.nn.functional.softmax(out.view(-1, self.natoms), 1).view(-1, self.out_size, self.natoms)
        
        return out


def make_env():
    """ Wrapped pong environment """
    env = gym.make("Pong-v4")
    env = ResizeAndScalePong(env)
    env = DerivativeEnv(env)
    env = NoopResetEnv(env, 20, 20)
    env = DoubleActionPong(env)
    return env


def main(args):
    """ Main function that prepares a model, extension dictionary and start
    training """
    env = make_env()
    state_shape = env.observation_space.shape
    state_shape = (4, state_shape[1], state_shape[2])
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
    )


    agent = RainBow(valuenet, act_size, extensions,
                    args.buffer_capacity, state_shape, state_dtype)

    if args.eval_mode:

        agent.load_state_dict(torch.load("models/RainBow_1820000_05.333.b")["model"])

    optimizer = torch.optim.Adam(valuenet.parameters(), lr=args.lr)
    agent.to(args.device)
    Trainer(args, agent, optimizer, env)()


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Rainbow')
    parser.add_argument("--n-iterations", type=int, default=50000000,
                        help="Number of training iterations")
    parser.add_argument("--start-update", type=int, default=500,
                        help="Number of iterations until starting to update")

    # ----------------------- Hyperparameters -----------------------
    parser.add_argument("--batch-size", type=int, default=32,
                        help="Batch size of each update in training")
    parser.add_argument("--gamma", type=float, default=0.99,
                        help="Discount Factor")
    parser.add_argument("--lr", type=float, default=0.0001,
                        help="Learning rate")
    parser.add_argument("--device", type=str, default="cuda",
                        help="Torch device")
    parser.add_argument("--target-update-period", type=int, default=1000,
                        help="Target network updating period")
    parser.add_argument("--buffer-capacity", type=int, default=10000,
                        help="Replay buffer capacity")
    parser.add_argument("--epsilon-init", type=float, default=1,
                        help="Initial value of the epsilon")
    parser.add_argument("--epsilon-min", type=float, default=0.01,
                        help="Minimum value of the epsilon")
    parser.add_argument("--epsilon-decay", type=float, default=0.85,
                        help="Epsilon decay rate for exponential decaying")
    parser.add_argument("--epsilon-range", type=int, default=150,
                        help="Epsilon decaying range for linear decay")
    parser.add_argument("--clip-grad", action="store_true", default=True,
                        help="Gradient Clip between -1 and 1. Default: No")

    # ----------------------- Extensions -----------------------
    parser.add_argument("--no-double", action="store_true", default=True,
                        help="Disable double DQN extension")
    parser.add_argument("--no-dueling", action="store_true", default=True,
                        help="Disable dueling DQN extension")

    parser.add_argument("--no-noisy", action="store_true", default=True,
                        help="Disable noisy layers")
    parser.add_argument("--noisy-std", type=float, default=0.5,
                        help="Initial std for noisy layers")

    parser.add_argument("--no-prioritized", action="store_true", default=True,
                        help="Disable prioritized replay buffer")
    parser.add_argument("--alpha", type=float, default=0.6,
                        help="Prioritization exponent")
    parser.add_argument("--beta-init", type=float, default=0.4,
                        help="Prioritization exponent")

    parser.add_argument("--n-steps", type=int, default=1,
                        help="Number of steps for bootstrapping")

    parser.add_argument("--no-dist", action="store_true", default=True,
                        help="Disable distributional DQN extension")
    parser.add_argument("--vmin", type=float, default=-10,
                        help="Minimum value for distributional DQN extension")
    parser.add_argument("--vmax", type=float, default=10,
                        help="Maximum value for distributional DQN extension")
    parser.add_argument("--natoms", type=int, default=51,
                        help="Number of atoms in distributional DQN extension")

    # ----------------------- Miscelenious -----------------------
    parser.add_argument("--eval-period", type=int, default=20000,
                        help="Evaluation period in terms of iteration")
    parser.add_argument("--eval-episode", type=int, default=3,
                        help="Number of episodes to evaluate")
    parser.add_argument("--save-model", action="store_true", default=False,
                        help="If given most successful models so far will be saved")
    parser.add_argument("--eval-mode", action="store_true", default=False,
                        help="Demonstration mode")
    parser.add_argument("--model-dir", type=str, default="models/",
                        help="Directory to save models")
    parser.add_argument("--write-period", type=int, default=500,
                        help="Writer period")

    args = parser.parse_args()

    main(args)
