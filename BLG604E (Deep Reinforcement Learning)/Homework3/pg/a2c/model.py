import torch
import numpy as np
from collections import namedtuple
from datetime import datetime

from pg.common import PrintWriter


class A2C(torch.nn.Module):
    """ Advantage Actor Critic agent. Learning is performed using vectorized
    environments.
        Arguments:
            - network: A neural network that includes both the value and the
            policy networks (possibly share a few layers).

    """

    Transition = namedtuple("Transition",
                            "reward done log_prob value entropy")

    # List contains a list of Transitions
    Rollout = namedtuple("Rollout", "list target_value")

    def __init__(self, network):
        super().__init__()
        self.network = network
        self.writer = PrintWriter(flush=True)

    def forward(self, state, *args, **kwargs):
        """ For the given state and all the other positional and keyword
        arguments calculate a distribution and sample action from it.
        Return the action, log probability of observing that sampled action,
        the value of the state, entropy, and remainder of the network's output.
            Arguments:
                - state: Input tensor
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
        return (action, log_prob, value, entropy, gru_hx)

    def to_torch(self, array, device):
        """ Helper method that transforms numpy array to a torch tensor """
        return torch.from_numpy(array).to(device).float()

    def accumulate_gradient(self, rollout, gamma, beta=0.0):
        """ Calculate a loss value for all the transitions in the rollout and
        backpropagate the loss. Every gradient is accumulated (by default)
            Arguments:
                - rollout: A Rollout instance (defined on the top of the class)
                that holds transitions.
                - gamma: Discount factor
                - beta: Entropy constant in the loss calculation
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

    def learn(self, args, opt, vecenv):
        """ The learner function is similar to sklearn API. In a loop, obtain
        rollouts in parallel (via vecenv) and perform gradient calculations to
        update the network parameters.
            Arguments:
                - args: Parsed command line arguments
                - opt: The optimizer to update the network parameters
                - vecenv: Synchronized multi environements
        """
        eps_reward = np.zeros((args.nenv, 1))  # Used to keep track of
        # cummulative rewards in multiple environments

        eps_reward_list = []  # Used to keep track of episodic reward of the
        # completed episodes from all the environments in ParallelEnv

        # Initilize the hidden vector
        gru_hx = torch.zeros(args.nenv, args.hidden_size).to(args.device)

        # Initilize and start the vecenv (Please check the implementation!!!!)
        states = vecenv.reset()
        for ix in range(0, args.n_iteration, args.n_step * args.nenv):
            gru_hx = gru_hx.detach()
            rollout_list = []

            for jx in range(0, args.n_step * args.nenv, args.nenv):
                #  /$$$$$$$$ /$$$$$$ /$$       /$$
                # | $$_____/|_  $$_/| $$      | $$
                # | $$        | $$  | $$      | $$
                # | $$$$$     | $$  | $$      | $$
                # | $$__/     | $$  | $$      | $$
                # | $$        | $$  | $$      | $$
                # | $$       /$$$$$$| $$$$$$$$| $$$$$$$$
                # |__/      |______/|________/|________/
                raise NotImplementedError

                # Don't forget to fill eps_reward_list with the episodic
                # rewards. Check if an environment is terminated and if so,
                # append it's cummulative reward (episodic reward) to the
                # eps_reward_list.

                if (ix + jx + args.nenv) % args.write_period == 0:
                    self.writer([
                        self.writer.Column(
                            "Iteration", "{:8}", (ix + jx + args.nenv)),
                        self.writer.Column("Reward", "{:6.2f}", np.mean(
                            eps_reward_list[-args.log_window:])),
                        self.writer.Column(
                            "Episode", "{:8}", len(eps_reward_list)),
                        self.writer.Column("Time", "{}", str(datetime.now())),
                    ])

            # End of a parallel rollout

            #  /$$$$$$$$ /$$$$$$ /$$       /$$
            # | $$_____/|_  $$_/| $$      | $$
            # | $$        | $$  | $$      | $$
            # | $$$$$     | $$  | $$      | $$
            # | $$__/     | $$  | $$      | $$
            # | $$        | $$  | $$      | $$
            # | $$       /$$$$$$| $$$$$$$$| $$$$$$$$
            # |__/      |______/|________/|________/
            raise NotImplementedError

    def save(self, path):
        """ Save the model parameters """
        torch.save(self.state_dict(), path)

    def test(self, path, envmaker, n_episodes=5, device="cpu"):
        """ Evaluate the agent loaded from the given path n_episodes many
        times and return the average episodic reward.
            Arguments:
                - path: Path to the model parameters
                - envmaker: Environment creating function
                - n_episodes: Number of episodes to evaluate (default: 5)
                - device: Torch device (default: cpu)

        This function will be used to evaluate the submitted model for Pong.
        """
        self.load_state_dict(torch.load(path))
        env = envmaker()
        episodic_rewards = []
        for ix in range(n_episodes):
            state = env.reset()
            done = False
            eps_reward = 0
            hx = torch.zeros(1, self.network.hidden_size).to(device).float()
            while not done:
                state = torch.from_numpy(state).unsqueeze(0).to(device).float()
                logit, value, hx = self.network(state, hx)
                action = torch.distributions.categorical.Categorical(
                    logits=logit)
                action = action.sample().item()
                state, reward, done, _ = env.step(action)
                eps_reward += reward
            episodic_rewards.append(eps_reward)
        return np.mean(episodic_rewards)