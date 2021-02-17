import torch
import numpy as np
from copy import deepcopy
from collections import namedtuple, deque
from functools import reduce

from .model import RainBow
from dqn.common import linear_annealing, exponential_annealing, PrintWriter
from dqn.dqn.train import Trainer as BaseTrainer


class Trainer(BaseTrainer):
    """ Training class that organize evaluation, update, and transition
    gathering.
        Arguments:
            - args: Parser arguments
            - agent: RL agent object
            - opt: Optimizer that optimizes agent's parameters
            - env: Gym environment
    """

    def __init__(self, args, agent, opt, env):

        super().__init__(args, agent, opt, env)
        # You can use this section to initialize beta and later increase it
        # incrementally in the update

    def update(self, ix):
        """ One step updating function. Update the agent in training mode.
        - clip gradient if "clip_grad" is given in args
        - keep track of td loss
        - Update target network
        If the prioritized buffer is active:
            - Take the weighted average of the loss with weights from
            the prioritized buffer
            - Update priorities of the sampled transitions
        If noisy-net is active:
            - reset noise for valuenet and targetnet (another reset for
            double Q-learning)
        Check for the training index "ix" to start the update.
            Arguments:
                - ix: Training iteration
        """
        self.agent.train()

        if ix > self.args.start_update:

            batch_size = self.args.batch_size

            self.opt.zero_grad()

            if not self.args.no_prioritized:
                transitions, selected_ids, weights = self.agent.buffer.sample(batch_size, self.args.beta_init)
                td_loss = self.agent.loss(transitions, self.args.gamma) # array of losses
                weights = torch.tensor(weights).to(self.args.device)

                td_loss[td_loss<0] = 0
                
                prios = td_loss.detach().cpu().numpy() + 1e-6


                td_loss = torch.mean(td_loss * weights)

                self.agent.buffer.update_priority(selected_ids, prios)

            else:
                transitions = self.agent.buffer.sample(batch_size)
                td_loss = self.agent.loss(transitions, self.args.gamma).mean()

            self.td_loss.append(td_loss.item())
            
            td_loss.backward()

            if self.args.clip_grad:
                for param in self.agent.valuenet.parameters():
                    param.grad.data.clamp_(-1, 1)

            self.opt.step()

            if ix % self.args.target_update_period == 0:
                self.agent.update_target()

            if not self.args.no_noisy:
                self.agent.valuenet.head.reset_noise()
                self.agent.targetnet.head.reset_noise()


    def __iter__(self):
        """ Multi-step transition generator """

        done = False

        ep_rew = 0

        n_steps = self.args.n_steps
        curr_step = 0

        while not done:

            state = self.state
            n_step_reward = 0

            for curr_step in range(n_steps):

                if self.args.no_noisy:
                    action = self.agent.e_greedy_policy(state, self.epsilon)
                else:
                    action = self.agent.greedy_policy(state)

                if (curr_step==0):
                    first_action = action

                next_state, reward, done, _ = self.env.step(action)

                return_done = done

                ep_rew += reward

                n_step_reward += reward * (self.args.gamma ** (curr_step+1))

                if done:
                    self.state = self.env.reset()
                    self.train_rewards.append(ep_rew)
                    ep_rew = 0
                    done = False
                    self.epsilon = next(self.epsilon_iterator)
                    break

                else:
                    self.state = next_state

            curr_step = 0

            yield self.agent.Transition(state, first_action, n_step_reward, next_state, return_done)
