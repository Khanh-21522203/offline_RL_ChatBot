from torch import nn
from typing import List
import torch

class MLP(nn.Module):
    """Defines a standard Multi Layer Perceptron"""

    def __init__(self, input_dim: int, hidden_dim: List[int], output_dim: int) -> None:
        
        nn.Module.__init__(self)
        # create mlp
        sizes = [input_dim] + hidden_dim + [output_dim]
        mods = sum(
            [
                [nn.Linear(i, j), nn.ReLU(inplace=True)]
                for i, j in zip(sizes, sizes[1:])
            ],
            [],
        )
        mods = mods[:-1]
        self.mlp = nn.Sequential(*mods)
        self._mlp_hyperparams = {
            "input_dim": input_dim,
            "output_dim": output_dim,
            "hidden_dim": hidden_dim,
        }

    def forward(self, x):
        return self.mlp.forward(x)

    def log_local_hyperparams(self):
        return self._mlp_hyperparams

    def log_local_epoch(self):
        return self.mlp.state_dict()
    
    
class DoubleQCritic(nn.Module):
    """Critic network, employes double Q-learning."""

    def __init__(self, state_dim, action_dim, hidden_dim: List[int]) -> None:
        nn.Module.__init__(self)
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.Q1 = MLP(self.state_dim + self.action_dim, hidden_dim, 1)
        self.Q2 = MLP(self.state_dim + self.action_dim, hidden_dim, 1)

    def double_q(self, state, action):
        state_action = torch.cat([state, action], dim=-1)
        q1, q2 = self.Q1(state_action), self.Q2(state_action)
        return q1, q2

    def forward(self, state, action):
        return torch.min(*self.double_q(state, action))