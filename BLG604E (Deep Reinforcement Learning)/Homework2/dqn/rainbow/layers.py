import torch
import numpy as np
import math


class HeadLayer(torch.nn.Module):
    """ Multi-function head layer. Structure of the layer changes depending on
    the "extensions" dictionary. If "noisy" is given Linear layers become
    Noisy Linear. If "dueling" is given dueling architecture is activated
    and lastly, if "distributional" is active, output shape is changed
    according to.
        Arguments:
            - in_size: Input size of the head layer
            - act_size: Action size that modifies Advantage and Q value shape
            - extensions: A dictionary that keeps extension information for
            Rainbow
            - hidden_size: Only used when "dueling" is active. Size of the
            hidden layers (both for value and advantage hidden layers).
    """

    def __init__(self, in_size, act_size, extensions, hidden_size=64):
        super().__init__()

        self.extensions = extensions

        out_size = act_size

        if extensions["distributional"]: # DIST ON
            out_size *= extensions["distributional"]["natoms"]
        
        if extensions["noisy"]: # NOISY ON
            noisy_std = extensions["noisy"]["init_std"]

        if not extensions["dueling"]: # DUELING OFF
            if extensions["noisy"]: # NOISY ON
                self.head = NoisyLinear(in_size, out_size, noisy_std)
            else: # NOISY OFF
                self.head = torch.nn.Linear(in_size, out_size)
               
        else: # DUELING ON

            if extensions["noisy"]: # NOISY ON
                self.vhead = torch.nn.Sequential(
                    NoisyLinear(in_size, hidden_size, noisy_std),
                    torch.nn.LeakyReLU(),
                    NoisyLinear(hidden_size, 1, noisy_std),
                )

                self.ahead = torch.nn.Sequential(
                    NoisyLinear(in_size, hidden_size, noisy_std),
                    torch.nn.LeakyReLU(),
                    NoisyLinear(hidden_size, out_size, noisy_std),
                )

            else: # NOISY OFF
                self.vhead = torch.nn.Sequential(
                    torch.nn.Linear(in_size, hidden_size),
                    torch.nn.LeakyReLU(),
                    torch.nn.Linear(hidden_size, 1),
                )

                self.ahead = torch.nn.Sequential(
                    torch.nn.Linear(in_size, hidden_size),
                    torch.nn.LeakyReLU(),
                    torch.nn.Linear(hidden_size, out_size),
                )


    def forward(self, x):

        if self.extensions["dueling"]: # DUELING ON
            advantage = self.ahead(x)
            return self.vhead(x) + self.ahead(x) - advantage.mean()

        else: # DUELING OFF
            return self.head(x)

    def reset_noise(self):
        """ Only used when "noisy" is active. Call reset_noise function of all
        child layers """
        try:
            self.head.reset_noise()
        except:
            pass

        try:
            for m in self.vhead.children():
                try:
                    m.reset_noise()
                except:
                    pass
        except:
            pass

        try:
            for m in self.ahead.children():
                try:
                    m.reset_noise()
                except:
                    pass
        except:
            pass

class NoisyLinear(torch.nn.Module):
    """Noisy linear module for NoisyNet.
    
    Attributes:
        in_features (int): input size of linear module
        out_features (int): output size of linear module
        std_init (float): initial std value
        weight_mu (nn.Parameter): mean value weight parameter
        weight_sigma (nn.Parameter): std value weight parameter
        bias_mu (nn.Parameter): mean value bias parameter
        bias_sigma (nn.Parameter): std value bias parameter
        
    """

    def __init__(self, in_features: int, out_features: int, std_init: float = 0.5):
        """Initialization."""
        super(NoisyLinear, self).__init__()
        
        self.in_features = in_features
        self.out_features = out_features
        self.std_init = std_init

        self.weight_mu = torch.nn.Parameter(torch.Tensor(out_features, in_features))
        self.weight_sigma = torch.nn.Parameter(
            torch.Tensor(out_features, in_features)
        )
        self.register_buffer(
            "weight_epsilon", torch.Tensor(out_features, in_features)
        )

        self.bias_mu = torch.nn.Parameter(torch.Tensor(out_features))
        self.bias_sigma = torch.nn.Parameter(torch.Tensor(out_features))
        self.register_buffer("bias_epsilon", torch.Tensor(out_features))

        self.reset_parameters()
        self.reset_noise()

    def reset_parameters(self):
        """Reset trainable network parameters (factorized gaussian noise)."""
        mu_range = 1 / math.sqrt(self.in_features)
        self.weight_mu.data.uniform_(-mu_range, mu_range)
        self.weight_sigma.data.fill_(
            self.std_init / math.sqrt(self.in_features)
        )
        self.bias_mu.data.uniform_(-mu_range, mu_range)
        self.bias_sigma.data.fill_(
            self.std_init / math.sqrt(self.out_features)
        )

    def reset_noise(self):
        """Make new noise."""
        epsilon_in = self.scale_noise(self.in_features)
        epsilon_out = self.scale_noise(self.out_features)

        # outer product
        self.weight_epsilon.copy_(epsilon_out.ger(epsilon_in))
        self.bias_epsilon.copy_(epsilon_out)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward method implementation.
        
        We don't use separate statements on train / eval mode.
        It doesn't show remarkable difference of performance.
        """
        return torch.nn.functional.linear(
            x,
            self.weight_mu + self.weight_sigma * self.weight_epsilon,
            self.bias_mu + self.bias_sigma * self.bias_epsilon,
        )
    
    @staticmethod
    def scale_noise(size: int) -> torch.Tensor:
        """Set scale to make noise (factorized gaussian noise)."""
        x = torch.FloatTensor(np.random.normal(loc=0.0, scale=1.0, size=size))

        return x.sign().mul(x.abs().sqrt())