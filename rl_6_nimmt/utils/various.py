import time
import torch
import logging
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np

logger = logging.getLogger(__name__)


def plot_grad_flow(named_parameters):
    """Plots the gradients flowing through different layers in the net during training.
    Can be used for checking for possible gradient vanishing / exploding problems.

    Usage: Plug this function in Trainer class after loss.backwards() as
    "plot_grad_flow(self.model.named_parameters())" to visualize the gradient flow"""
    ave_grads = []
    max_grads = []
    layers = []
    for n, p in named_parameters:
        if (p.requires_grad) and ("bias" not in n):
            layers.append(n)
            ave_grads.append(p.grad.abs().mean())
            max_grads.append(p.grad.abs().max())
    plt.bar(np.arange(len(max_grads)), max_grads, alpha=0.1, lw=1, color="c")
    plt.bar(np.arange(len(max_grads)), ave_grads, alpha=0.1, lw=1, color="b")
    plt.hlines(0, 0, len(ave_grads) + 1, lw=2, color="k")
    plt.xticks(range(0, len(ave_grads), 1), layers, rotation="vertical")
    plt.xlim(left=0, right=len(ave_grads))
    plt.ylim(bottom=-0.001, top=0.02)  # zoom in on the lower gradient regions
    plt.xlabel("Layers")
    plt.ylabel("average gradient")
    plt.title("Gradient flow")
    plt.grid(True)
    plt.legend(
        [Line2D([0], [0], color="c", lw=4), Line2D([0], [0], color="b", lw=4), Line2D([0], [0], color="k", lw=4)],
        ["max-gradient", "mean-gradient", "zero-gradient"],
    )


def compute_discounted_returns(rewards, gamma, dtype=torch.float, device=torch.device("cpu")):
    if isinstance(rewards, torch.Tensor):
        rewards = rewards.numpy()
    returns = []
    actual_return = 0.0
    for reward in rewards[::-1]:
        actual_return = reward + gamma * actual_return
        returns.insert(0, actual_return)
    returns = torch.tensor(returns).to(device, dtype)
    return returns


def timeit(method):
    def timed(*args, **kw):
        ts = time.time()
        result = method(*args, **kw)
        te = time.time()
        logger.info(f"{method.__name__}  {(te - ts) * 1000:2.2f} ms")
        return result

    return timed


def iter_flatten(iterable, max_depth=None):
    """ From http://rightfootin.blogspot.com/2006/09/more-on-python-flatten.html """
    it = iter(iterable)
    for e in it:
        if isinstance(e, (list, tuple, np.ndarray)) and (max_depth is None or max_depth > 0):
            for f in iter_flatten(e, None if max_depth is None else max_depth - 1):
                yield f
        else:
            yield e
