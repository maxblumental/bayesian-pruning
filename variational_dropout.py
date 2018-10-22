import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal


class SparseLinear(nn.Module):
    def __init__(self, in_features, out_features, bias=True):
        super(SparseLinear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.W = nn.Parameter(torch.Tensor(in_features, out_features))
        if bias:
            self.bias = nn.Parameter(torch.Tensor(out_features))
        else:
            self.register_parameter('bias', None)
        self.log_sigma2 = nn.Parameter(torch.ones(in_features, out_features).mul_(-25))
        self.sampler = Normal(0, 1)
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.W.size(1))
        self.W.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def forward(self, input):
        log_alpha = self.log_sigma2 - torch.log(self.W ** 2)
        alpha = log_alpha.clamp(-8, 8).exp()
        mu = input.matmul(self.W)
        si = input.pow(2).matmul(alpha.mul(self.W.pow(2))).add_(1e-8).sqrt_()
        activation = self.sampler.sample(mu.shape).to(self.W.device).mul(si).add_(mu)
        return activation + self.bias

    def kl_term(self):
        k1, k2, k3 = 0.63576, 1.8732, 1.48695;
        C = -k1
        log_alpha = self.log_sigma2 - torch.log(self.W ** 2)
        log_alpha.clamp(-8, 8)  # clip log alpha
        mdkl = k1 * torch.sigmoid(k2 + k3 * log_alpha) - 0.5 * torch.log1p(torch.exp(-log_alpha)) + C
        return -torch.sum(mdkl)


class Sgvb():
    def __init__(self, N, M):
        self.n = N
        self.m = M

    def kl_term(self, net):
        return sum([layer.kl_term() for layer in net.children()
                    if 'kl_term' in layer.__class__.__dict__])

    def ell(self, predictions, target):
        """
        Expected log-likelihood.
        """
        return -F.cross_entropy(predictions, target)

    def evaluate(self, predictions, targets, net):
        return self.n * 1.0 / self.m * self.ell(predictions, targets) - self.kl_term(net)
