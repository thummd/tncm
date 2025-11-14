import torch
from torch import nn
from torch import Tensor
import numpy as np

def log_standard_normal(x):
    log_p = -0.5 * torch.log(2.0 * torch.tensor(np.pi)) - 0.5 * x**2.0
    #log_p = torch.sum(log_p, dim=-1)
    log_p = torch.sum(log_p, dim=2)
    return log_p

class FlowPrior():
    def __init__(self, num_flows, latent_dim, hidden_dim):
        super(FlowPrior, self).__init__()

        L = latent_dim  # number of latents
        M = hidden_dim  # the number of neurons in scale (s) and translation (t) nets
        self.D = latent_dim//4
        #self.D = 16

        # scale (s) network
        nets = lambda: nn.Sequential(nn.Linear(self.D, M), nn.LeakyReLU(), nn.Linear(M, M), nn.LeakyReLU(), nn.Linear(M, self.D), nn.Tanh())

        # translation (t) network
        nett = lambda: nn.Sequential(nn.Linear(self.D, M), nn.LeakyReLU(), nn.Linear(M, M), nn.LeakyReLU(), nn.Linear(M, self.D))

        self.t = torch.nn.ModuleList([nett() for _ in range(num_flows)])
        self.s = torch.nn.ModuleList([nets() for _ in range(num_flows)])
        self.num_flows = num_flows

    def coupling(self, x, index, forward=True):
        (xa, xb) = torch.chunk(x, 2, 2)

        s = self.s[index](xa)
        t = self.t[index](xa)

        if forward:
            # yb = f^{-1}(x)
            yb = (xb - t) * torch.exp(-s)
        else:
            # xb = f(y)
            yb = torch.exp(s) * xb + t

        return torch.cat((xa, yb), 2), s

    def permute(self, x):
        return x.flip(1)

    def f(self, x):
        log_det_J, z = x.new_zeros(x.shape[0],x.shape[1]), x
        for i in range(self.num_flows):
            z, s = self.coupling(z, i, forward=True)
            z = self.permute(z)
            log_det_J = log_det_J - s.sum(dim=2)

        return z, log_det_J

    def f_inv(self, z):
        x = z
        for i in reversed(range(self.num_flows)):
            x = self.permute(x)
            x, _ = self.coupling(x, i, forward=False)

        return x

    def sample(self, n_sample, n_steps, device, generator=None):
        z = torch.randn((n_sample, n_steps, self.D*2), device=device, generator=generator)
        x = self.f_inv(z)
        return x

    def log_prob(self, x: Tensor) -> Tensor:
        z, log_det_J = self.f(x)
        log_p = log_standard_normal(z) + log_det_J
        return log_p