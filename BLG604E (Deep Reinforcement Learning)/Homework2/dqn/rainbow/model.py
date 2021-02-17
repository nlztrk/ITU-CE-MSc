import torch
import numpy as np
from copy import deepcopy
from collections import namedtuple

from dqn.replaybuffer.uniform import UniformBuffer
from dqn.replaybuffer.prioritized import PriorityBuffer
from dqn.base_dqn import BaseDQN


class RainBow(BaseDQN):
    """ Flexible Rainbow DQN implementations with droppable extensions.
        Arguments:
            - valunet: Neural network to estimate values
            - nact: Number of actions (and outputs)
            - extensions: A dictionary that keeps extension information
            - buffer_args: Remaning positional arguments to feed replay buffer
    """

    def __init__(self, valuenet, nact, extensions, *buffer_args):
        super().__init__(valuenet, nact)
        self.extensions = extensions

        if extensions["prioritized"]:
            self.buffer = PriorityBuffer(
                *buffer_args,
                alpha=extensions["prioritized"]["alpha"]
            )
        else:
            self.buffer = UniformBuffer(*buffer_args)
        
        self.nact = nact

    def greedy_policy(self, state, *args):
        """ The greedy policy that changes its behavior depending on the
        value of the "distributional" option in extensions dictionary
        """

        state = self.state_to_torch(state, "cuda")

        with torch.no_grad():
            values = self.valuenet.forward(state)
            values = values.view(1, self.nact, -1)

            if self.extensions["distributional"]:
                values = values.detach()
                values = self.expected_value(values)
                action = values.sum(2).max(1)[1].detach()[0].item()

            else:
                action = values.argmax().item()

        return action

    def loss(self, batch, gamma):
        """ Loss function that switches loss function depending on the value
        of the "distributional" option in extensions dictionary. Note that: For
        target value calculation "_next_action_network" should be used.
            Arguments:
                - batch: Transition object that keeps batch of sampled
                transitions
                - gamma: Discount Factor
        """
        if self.extensions["distributional"]:
            return self.distributional_loss(batch, gamma)
        return self.vanilla_loss(batch, gamma)

    def vanilla_loss(self, batch, gamma):
        """ MSE (L2, L1, or smooth L1) TD loss with double DQN extension in
        mind. Different than DQN loss, returning loss tensor is not averaged
        over the batch axis.
            Arguments:
                - batch: Transition object that keeps batch of sampled
                transitions
                - gamma: Discount Factor
        """
        
        loss = torch.nn.SmoothL1Loss(reduction="none")

        next_states = torch.Tensor(batch.next_state).to("cuda")
        states = torch.Tensor(batch.state).to("cuda")
        batch_size = states.shape[0]

        rewards = torch.Tensor(batch.reward).to("cuda").view(batch_size)
        actions = torch.Tensor(batch.action).to("cuda")
        terminals = torch.Tensor(batch.terminal).float().to("cuda")

        mult_terminals = 1 - terminals.clone().view(batch_size)
        
        ## TARGETNET
        next_state_values = torch.from_numpy(np.zeros(batch_size)).float().to("cuda")

        selected_action = self._next_action_network(next_states).argmax(dim=1, keepdim=True)
        next_state_values = self.targetnet(next_states).gather(1, selected_action).view(-1)

        next_state_values = mult_terminals * next_state_values
        ###

        expected_state_action_values = (next_state_values * gamma) + rewards

        ## VALUENET
        state_action_values = torch.from_numpy(np.zeros(batch_size)).float().to("cuda")

        actions = actions[:,0].view(-1,1).long()
        state_action_values = self.valuenet(states).gather(1, actions).view(-1)
        ###

        output = loss(state_action_values, expected_state_action_values)

        return output


    def expected_value(self, values):
        """ Return expectation of state-action values.
            Arguments:
                - values: Value tensor of distributional output (B, A, Z). B,
                A, Z denote batch, action, and atom respectively.
            Return:
                the expected value of shape (B, A)
        """
        vmin = self.extensions["distributional"]["vmin"]
        vmax = self.extensions["distributional"]["vmax"]
        natoms = self.extensions["distributional"]["natoms"]

        probs = torch.linspace(vmin, vmax, natoms).cuda()
        expected_next_value = values.mul(probs)

        return expected_next_value

    def distributional_loss(self, batch, gamma):
        """ Distributional RL TD loss with KL divergence (with Double
        Q-learning via "_next_action_network" at target value calculation).
        Different than DQN loss, returning loss tensor is not averaged over
        batch axis.
            Arguments:
                - batch: Transition object that keeps batch of sampled
                transitions
                - gamma: Discount Factor
        """
        vmin = self.extensions["distributional"]["vmin"]
        vmax = self.extensions["distributional"]["vmax"]
        natoms = self.extensions["distributional"]["natoms"]

        next_states = torch.Tensor(batch.next_state).to("cuda")
        states = torch.Tensor(batch.state).to("cuda")
        batch_size = states.shape[0]

        rewards = torch.Tensor(batch.reward).to("cuda").view(batch_size)
        actions = torch.Tensor(batch.action).long().to("cuda") 
        terminals = torch.Tensor(batch.terminal).float().to("cuda")

        delta_z = float(vmax - vmin) / (natoms - 1)
        support = torch.linspace(vmin, vmax, natoms).cuda()

        next_dist = self._next_action_network.forward(next_states).detach().mul(support)
        next_action = next_dist.sum(2).max(1)[1]
        next_action = next_action.unsqueeze(1).unsqueeze(1).expand(batch_size, 1, natoms)
        
        target_out = self.targetnet.forward(next_states).detach().mul(support)

        next_dist = target_out.gather(1, next_action).squeeze(1)
        rewards = rewards.unsqueeze(1).expand_as(next_dist)
        terminals = terminals.expand_as(next_dist)
        support = support.unsqueeze(0).expand_as(next_dist)

        Tz = rewards + (1 - terminals) * support * gamma
        Tz = Tz.clamp(min=vmin, max=vmax)
        b = (Tz - vmin) / delta_z
        l = b.floor().long()
        u = b.ceil().long()

        offset = torch.linspace(0, (batch_size - 1) * natoms, batch_size).long().unsqueeze(1).expand_as(next_dist).cuda()

        proj_dist = torch.zeros_like(next_dist, dtype=torch.float32).to("cuda")
        proj_dist.view(-1).index_add_(0, (offset + l).view(-1), (next_dist * (u.float() - b)).view(-1))
        proj_dist.view(-1).index_add_(0, (offset + u).view(-1), (next_dist * (b - l.float())).view(-1))
        
        dist = self.valuenet.forward(states)
        actions = actions.unsqueeze(1).expand(batch_size, 1, natoms)
        dist = dist.gather(1, actions).squeeze(1)
        dist.detach().clamp_(min=1e-3)

        loss = - (proj_dist * dist.log()).sum(1)

        return loss

    @property
    def _next_action_network(self):
        """ Return the network used for the next action calculation (Double
        Q-learning) """

        if self.extensions["double"]:
            return self.valuenet
        else:
            return self.targetnet

        
    def push_transition(self, transition):
        self.buffer.push(transition)
