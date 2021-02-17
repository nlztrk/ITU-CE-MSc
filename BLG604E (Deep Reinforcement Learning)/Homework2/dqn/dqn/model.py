import torch
import numpy as np

from dqn.replaybuffer.uniform import UniformBuffer
from dqn.base_dqn import BaseDQN


class DQN(BaseDQN):
    """ Deep Q Network that uses the target network and uniform replay buffer.
        Arguments:
            - valunet: Neural network to estimate values
            - nact: Number of actions (and outputs)
            - buffer_args: Remaning positional arguments to feed replay buffer
    """

    def __init__(self, valuenet, nact, *buffer_args):
        super().__init__(valuenet, nact)
        self.buffer = UniformBuffer(*buffer_args)

    def push_transition(self, transition):
        self.buffer.push(transition)

    def loss(self, batch, gamma):
        """ DQN loss that uses the target network to estimate target
        values.
            Arguments:
                - batch: Batch of transition as Transition namedtuple defined
                in BaseDQN class
                - gamma: Discount factor
            Return:
                td_error tensor: MSE loss (L1, L2 or smooth L1) of the target
                and predicted values
        """
        
        loss = torch.nn.MSELoss(reduction='mean')

        next_states = torch.Tensor(batch.next_state).to("cuda")
        states = torch.Tensor(batch.state).to("cuda")
        batch_size = states.shape[0]

        rewards = torch.Tensor(batch.reward).to("cuda").view(batch_size)
        actions = torch.Tensor(batch.action).to("cuda")
        terminals = torch.Tensor(batch.terminal).float().to("cuda")

        mult_terminals = terminals.clone().view(batch_size)

        mult_terminals[mult_terminals==0.] = 2.
        mult_terminals[mult_terminals==1.] = 0.
        mult_terminals[mult_terminals==2.] = 1.
        
        ## TARGETNET
        next_state_values = torch.from_numpy(np.zeros(batch_size)).float().to("cuda")
        next_state_values, _ = torch.max(self.targetnet(next_states), dim=1)        
        next_state_values = mult_terminals * next_state_values           
        ###

        expected_state_action_values = (next_state_values * gamma) + rewards

        ## VALUENET
        state_action_values = torch.from_numpy(np.zeros(batch_size)).float().to("cuda")
        actions = actions[:,0].view(-1,1).long()
        state_action_values = self.valuenet(states).gather(1, actions).view(-1)

        output = loss(state_action_values, expected_state_action_values)

        return output

