import numpy as np
import torch
import torch.nn.modules as nn
import torch.nn.functional as F

LOG_STD_MAX = 2
LOG_STD_MIN = -20
RAND_SEED = 0
LAYER_SIZE = 128

# Actor Net
class Actor(nn.Module):
    def __init__(self, input_dim, output_dim, output_bound):
        super().__init__()
        torch.manual_seed(RAND_SEED)
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.output_bound = output_bound

        self.net = nn.Sequential(
            nn.Linear(self.input_dim, 2*LAYER_SIZE),
            nn.PReLU(2*LAYER_SIZE),
            nn.Linear(2*LAYER_SIZE, LAYER_SIZE),
            nn.PReLU(LAYER_SIZE),
            nn.Linear(LAYER_SIZE, self.output_dim),
            nn.Tanh()
        )

    def forward(self, ob_goals):
        actions = self.output_bound * self.net(ob_goals)
        return actions


# Critic Net
class Critic(nn.Module):
    def __init__(self, input_dim, output_dim):
        torch.manual_seed(RAND_SEED)
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim

        self.net = nn.Sequential(
            nn.Linear(self.input_dim, 2*LAYER_SIZE),
            nn.PReLU(2*LAYER_SIZE),
            nn.Linear(2*LAYER_SIZE, LAYER_SIZE),
            nn.PReLU(LAYER_SIZE),
            nn.Linear(LAYER_SIZE, self.output_dim)
        )

    def forward(self, ob_goals, actions):
        # concatnate the observations and actions
        input = torch.cat([ob_goals, actions] ,dim=1)
        # put the input data into network, predict the q value
        q = self.net(input)
        return q