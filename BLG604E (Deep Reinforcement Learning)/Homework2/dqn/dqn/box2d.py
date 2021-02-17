import torch
import numpy as np
import gym
import argparse
import sys

from dqn.dqn.model import DQN
from dqn.common import PrintWriter
from dqn.dqn.train import Trainer


class ValueNet(torch.nn.Module):
    """ Fully connected neural network to estimate values.
        Arguments:
            - in_size: Size of a state
            - out_size: Action size
    """
    def __init__(self, in_size, out_size):
        super(ValueNet, self).__init__()

        self.value_net = torch.nn.Sequential(
                torch.nn.Linear(in_size, 128),
                torch.nn.ReLU(),
                torch.nn.Linear(128, 128),
                torch.nn.ReLU(),
                torch.nn.Linear(128, out_size),
            )

    def forward(self, input):
        return self.value_net(input)



def make_env(envname):
    """ Environment creating function """
    env = gym.make(args.envname)
    return env


def main(args):
    """ The main function that prepares a model and starts training  """
    env = make_env(args.envname)
    env._max_episode_steps = args.max_episode_len
    state_shape = env.observation_space.shape
    state_dtype = env.observation_space.dtype

    act_size = env.action_space.n
    valuenet = ValueNet(state_shape[0], act_size)
    agent = DQN(valuenet, act_size,
                args.buffer_capacity, state_shape, state_dtype)
    optimizer = torch.optim.Adam(valuenet.parameters(), lr=args.lr)
    Trainer(args, agent, optimizer, env)()


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='DQN')
    parser.add_argument("--envname", type=str,
                        default="LunarLander-v2",
                        help="Name of the environment")
    parser.add_argument("--n-iterations", type=int, default=150000,
                        help="Number of training iterations")
    parser.add_argument("--start-update", type=int, default=500,
                        help="Number of iterations until starting to update")
    parser.add_argument("--max-episode-len", type=int, default=1000,
                        help="Maximum length of an episode before termination")

    parser.add_argument("--batch-size", type=int, default=32,
                        help="Batch size of each update in training")
    parser.add_argument("--gamma", type=float, default=0.99,
                        help="Discount Factor")
    parser.add_argument("--lr", type=float, default=3e-4,
                        help="Learning Rate")
    parser.add_argument("--device", type=str, default="cuda",
                        help="Torch device")
    parser.add_argument("--target-update-period", type=int, default=10,
                        help="Target network updating period")
    parser.add_argument("--buffer-capacity", type=int, default=10000,
                        help="Replay buffer capacity")
    parser.add_argument("--epsilon-init", type=float, default=1,
                        help="Initial value of the epsilon")
    parser.add_argument("--epsilon-min", type=float, default=0.01,
                        help="Minimum value of the epsilon")
    parser.add_argument("--epsilon-decay", type=float, default=.995,
                        help="Epsilon decay rate for exponential decaying")
    parser.add_argument("--epsilon-range", type=int, default=None,
                        help="Epsilon decaying range for linear decay")
    parser.add_argument("--clip-grad", action="store_true", default=True,
                        help="Gradient Clip between -1 and 1. Default: No")

    parser.add_argument("--eval-period", type=int, default=1000,
                        help="Evaluation period in terms of iteration")
    parser.add_argument("--eval-episode", type=int, default=5,
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
