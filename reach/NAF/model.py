import torch
import torch.nn as nn
from torch.optim import Adam
from torch.autograd import Variable
import torch.nn.functional as F

LAYER_SIZE = 128

class Policy(nn.Module):
    def __init__(self, numInputs, numOutputs):
        super(Policy, self).__init__()
        self.numOutputs = numOutputs
        self.numInputs = numInputs

        self.V = nn.Sequential(
            nn.Linear(numInputs, 2*LAYER_SIZE),
            nn.PReLU(2*LAYER_SIZE),
            nn.Linear(2*LAYER_SIZE, LAYER_SIZE),
            nn.PReLU(LAYER_SIZE),
            nn.Linear(LAYER_SIZE, 1))

        self.mu = nn.Sequential(
            nn.Linear(numInputs, 2*LAYER_SIZE),
            nn.PReLU(2*LAYER_SIZE),
            nn.Linear(2*LAYER_SIZE, LAYER_SIZE),
            nn.PReLU(LAYER_SIZE),
            nn.Linear(LAYER_SIZE, numOutputs),
            nn.Tanh()
            )

        self.L = nn.Sequential(
            nn.Linear(numInputs, LAYER_SIZE),
            nn.PReLU(LAYER_SIZE),
            nn.Linear(LAYER_SIZE, numOutputs*numOutputs),
            nn.PReLU(numOutputs*numOutputs),
            nn.Linear(numOutputs*numOutputs, numOutputs*numOutputs))

        self.trilMask = torch.tril(torch.ones(numOutputs, numOutputs), diagonal=-1).unsqueeze(0)
        self.diagMask = torch.diag(torch.ones(numOutputs, numOutputs)).unsqueeze(0)


    def forward(self, inputs):
        x, u = inputs

        V = self.V(x)
        mu = self.mu(x)

        Q = None
        if u is not None:
            L = self.L(x).view(-1, self.numOutputs, self.numOutputs)
            L = L * self.trilMask.expand_as(L) + torch.exp(L) * self.diagMask.expand_as(L)
            P = torch.bmm(L, L.transpose(2, 1))
            uMu = (u-mu).unsqueeze(2)
            A = -0.5*torch.bmm(torch.bmm(uMu.transpose(2,1), P), uMu)[:, :, 0]
            Q = A+V

        return mu, Q, V