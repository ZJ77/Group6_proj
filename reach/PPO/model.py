import torch
import torch.nn as nn
from torch.distributions import MultivariateNormal

LAYER_SIZE = 128

class ActorCritic(nn.Module):
    def __init__(self, state_dim, action_dim, action_std):
        super(ActorCritic, self).__init__()
        # action mean range -1 to 1
        self.actor = nn.Sequential(
            nn.Linear(state_dim, 2*LAYER_SIZE),
            nn.PReLU(),
            nn.Linear(2*LAYER_SIZE, LAYER_SIZE),
            nn.PReLU(),
            nn.Linear(LAYER_SIZE, action_dim),
            nn.Tanh()
        )
        # critic
        self.critic = nn.Sequential(
            nn.Linear(state_dim, 2*LAYER_SIZE),
            nn.PReLU(),
            nn.Linear(2*LAYER_SIZE, LAYER_SIZE),
            nn.PReLU(),
            nn.Linear(LAYER_SIZE, 1)
        )
        self.action_var = torch.full((action_dim,), action_std * action_std)

    def forward(self):
        raise NotImplementedError

    def act(self, state):
        action_mean = self.actor(state)
        cov_mat = torch.diag(self.action_var)

        dist = MultivariateNormal(action_mean, cov_mat)
        action = dist.sample()
        ac_logprob = dist.log_prob(action)

        return action.detach(), ac_logprob.detach()

    def eval(self, state, action):
        action_mean = self.actor(state)

        action_var = self.action_var.expand_as(action_mean)
        cov_mat = torch.diag_embed(action_var)

        dist = MultivariateNormal(action_mean, cov_mat)

        ac_logprobs = dist.log_prob(action)
        dist_entropy = dist.entropy()
        state_value = self.critic(state)
        state_value = torch.squeeze(state_value)

        return ac_logprobs, state_value, dist_entropy