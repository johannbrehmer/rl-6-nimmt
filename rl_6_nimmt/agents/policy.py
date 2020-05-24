import torch
from torch import nn
from torch.distributions import Categorical
import numpy as np
import logging

from .base import Agent
from ..utils.nets import MultiHeadedMLP
from ..utils.various import compute_discounted_returns

logger = logging.getLogger(__name__)


class MaskedReinforceAgent(Agent):
    def __init__(
        self,
        env,
        gamma=0.99,
        optim_kwargs=None,
        history_length=None,
        dtype=torch.float,
        device=torch.device("cpu"),
        hidden_sizes=(100,),
        activation=nn.ReLU(),
        r_factor=1.0,
        actor_weight=1.0,
        entropy_weight=0.0,
        *args,
        **kwargs
    ):
        super().__init__(env, gamma, optim_kwargs, history_length, dtype, device)

        self.r_factor = r_factor
        self.actor_weight = actor_weight
        self.entropy_weight = entropy_weight

        # NN that calculates the policy (actor) and estimates Q (critic)
        self.actor = MultiHeadedMLP(self.state_length, hidden_sizes=hidden_sizes, head_sizes=(self.num_actions,), activation=activation, head_activations=(None,))
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, state, legal_actions, **kwargs):
        # Let the actor pick action probabilities and the critic guess the expected reward V(s_t)
        (probs,) = self.actor(state)
        probs = probs[legal_actions]
        probs = self.softmax(probs)

        # Sample action from these probabilities
        cat = Categorical(probs)
        action_id = cat.sample()
        log_prob = cat.log_prob(action_id)
        entropy = cat.entropy()
        action = legal_actions[action_id]

        return int(action.item()), {"log_prob": log_prob, "entropy": entropy}

    def learn(self, state, reward, action, done, next_state, next_reward, episode_end, num_episode, *args, **kwargs):
        # Memorize step
        self.history.store(log_prob=kwargs["log_prob"], reward=reward * self.r_factor, entropy=kwargs["entropy"])

        # No further steps after each step
        if not episode_end or not self.training:
            return np.zeros(3)

        # Gradient updates
        losses = self._train()

        # Reset memory for next episode
        self.history.clear()

        return losses

    def _train(self):
        # Roll out last episode
        rollout = self.history.rollout()
        n = len(self.history)
        log_probs = torch.stack(rollout["log_prob"], dim=0)
        entropies = torch.stack(rollout["entropy"], dim=0)
        returns = compute_discounted_returns(rollout["reward"], self.gamma).to(self.device, self.dtype)

        # Compute loss
        discounts = torch.exp(np.log(self.gamma) * torch.linspace(0, n - 1, n))
        discounts = discounts.to(self.device, self.dtype)
        actor_loss = -torch.sum(discounts * returns * log_probs)

        # Entropy regularization to incentivize exploration
        if entropies is not None:
            entropy_loss = -torch.sum(entropies)
        else:
            entropy_loss = torch.tensor(0.0)

        # Gradient update
        self._gradient_step(self.actor_weight * actor_loss + self.entropy_weight * entropy_loss)

        return np.array([actor_loss.item(), 0.0, entropy_loss.item()])

    def _gradient_step(self, loss):
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
