""" MLP implementation """
import math
import torch
import torch.nn.functional as F
from torch.autograd import Variable
from torch import nn
import logging
from functools import partial

logger = logging.getLogger(__file__)

# from https://github.com/Shmuma/ptan/blob/master/samples/rainbow/lib/dqn_model.py
class NoisyLinear(nn.Linear):
    def __init__(self, in_features, out_features, sigma_init=0.5, bias=True):
        super(NoisyLinear, self).__init__(in_features, out_features, bias=bias)
        self.sigma_weight = nn.Parameter(torch.Tensor(out_features, in_features).fill_(sigma_init))
        self.register_buffer("epsilon_weight", torch.zeros(out_features, in_features))
        if bias:
            self.sigma_bias = nn.Parameter(torch.Tensor(out_features).fill_(sigma_init))
            self.register_buffer("epsilon_bias", torch.zeros(out_features))
        self.reset_parameters()

    def reset_parameters(self):
        std = math.sqrt(3 / self.in_features)
        nn.init.uniform(self.weight, -std, std)
        nn.init.uniform(self.bias, -std, std)

    def forward(self, input):
        torch.randn(self.epsilon_weight.size(), out=self.epsilon_weight)
        bias = self.bias
        if bias is not None:
            torch.randn(self.epsilon_bias.size(), out=self.epsilon_bias)
            bias = bias + self.sigma_bias * Variable(self.epsilon_bias)
        return F.linear(input, self.weight + self.sigma_weight * Variable(self.epsilon_weight), bias)


class NoisyFactorizedLinear(nn.Linear):
    """
    NoisyNet layer with factorized gaussian noise
    N.B. nn.Linear already initializes weight and bias to
    """

    def __init__(self, in_features, out_features, sigma_init=0.4, bias=True):
        super(NoisyFactorizedLinear, self).__init__(in_features, out_features, bias=bias)
        sigma_init = sigma_init / math.sqrt(in_features)
        self.sigma_weight = nn.Parameter(torch.Tensor(out_features, in_features).fill_(sigma_init))
        self.register_buffer("epsilon_input", torch.zeros(1, in_features))
        self.register_buffer("epsilon_output", torch.zeros(out_features, 1))
        if bias:
            self.sigma_bias = nn.Parameter(torch.Tensor(out_features).fill_(sigma_init))

    def forward(self, input):
        torch.randn(self.epsilon_input.size(), out=self.epsilon_input)
        torch.randn(self.epsilon_output.size(), out=self.epsilon_output)

        func = lambda x: torch.sign(x) * torch.sqrt(torch.abs(x))
        eps_in = func(self.epsilon_input)
        eps_out = func(self.epsilon_output)

        bias = self.bias
        if bias is not None:
            bias = bias + self.sigma_bias * Variable(eps_out.t())
        noise_v = Variable(torch.mul(eps_in, eps_out))
        return F.linear(input, self.weight + self.sigma_weight * noise_v, bias.view(self.weight.shape[0]))


class CNN(nn.Module):
    # WIP not happy with it yet.
    def __init__(
        self,
        in_channels,
        hidden_sizes,
        out_channels,
        activation=nn.ReLU(),
        reducer=nn.MaxPool2d(kernel_size=2, stride=2, padding=0),
        kernel_size=3,
        stride=1,
        padding=1,
    ):
        super().__init__()
        self.sizes = (in_channels,) + hidden_sizes + (out_channels,)
        layers = list()
        for i, s in enumerate(self.sizes[:-1]):
            s_next = self.sizes[i + 1]
            block = [nn.Conv2d(s, s_next, kernel_size=kernel_size, stride=stride, padding=padding), activation, reducer]
            layers += block
        self.latent_net = nn.Sequential(*layers)

    def linear_out_size(self, h, w):
        num_layers = len(self.sizes)
        out_features = self.sizes[-1]
        conv_h = int(h * 0.5 ** num_layers)
        conv_w = int(w * 0.5 ** num_layers)
        out_size = conv_h * conv_w * out_features
        return out_size

    def forward(self, x):
        return self.latent_net(x)


class MultiHeadedMLP(nn.Module):
    """ MLP with multiple heads """

    def __init__(self, input_size, hidden_sizes, head_sizes, activation, head_activations, linear=nn.Linear, init_sigma=1.0):
        super().__init__()
        # print(hidden_sizes)
        if linear != nn.Linear:
            logger.info(f"Creating noisy network with init sigma {init_sigma}")
            linear = partial(
                linear, sigma_init=init_sigma
            )  # partial takes care of the inital noise param, cleans up rest of the code, by avoiding if/else all over

        layers = [linear(input_size, hidden_sizes[0]), activation]
        for last_hidden, hidden in zip(hidden_sizes[:-1], hidden_sizes[1:]):
            logger.info(f"Creating Linear layer: {last_hidden}->{hidden}")
            l = linear(last_hidden, hidden)
            layers.append(l)
            layers.append(activation)
        self.latent_net = nn.Sequential(*layers)

        self.head_nets = nn.ModuleList()
        for head_size, head_activation in zip(head_sizes, head_activations):
            logger.info(f"Creating Linear head layer: {hidden_sizes[-1]}->{head_size}")
            layers = [linear(hidden_sizes[-1], head_size)]

            if head_activation is not None:
                layers.append(head_activation)
            self.head_nets.append(nn.Sequential(*layers))

    def forward(self, inputs):
        latent = self.latent_net(inputs)
        outputs = [head(latent) for head in self.head_nets]
        return outputs


class DuellingDQNNet(nn.Module):
    def __init__(self, input_size, hidden_sizes, out_size, activation, linear=nn.Linear, init_sigma=1.0):

        super().__init__()
        self.mlp = MultiHeadedMLP(input_size, hidden_sizes, (1, out_size), activation, (None, None), linear=linear, init_sigma=init_sigma)

    def forward(self, inputs):
        V, A = self.mlp(inputs)
        outputs = V + (A - A.mean(keepdims=True, dim=-1))
        return [outputs]
