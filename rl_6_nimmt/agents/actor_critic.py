import torch
from torch import nn
from torch.distributions import Categorical
import numpy as np
import logging

from .base import Agent
from ..utils.nets import MultiHeadedMLP
from ..utils.various import compute_discounted_returns, iter_flatten
from ..utils.replay_buffer import SequentialHistory
from ..utils.preprocessing import SechsNimmtStateNormalization

logger = logging.getLogger(__name__)


class BatchedActionValueActorCriticAgent(Agent):
    """
    Simple actor-critic agent for discrete action spaces.

    The actor implements the policy pi(a|s). The critic estimates the state-action value q(s, a). The base A-C class does not yet implement any training algorithm.
    """

    def __init__(
        self,
        env=None,
        gamma=0.99,
        optim_kwargs=None,
        history_length=None,
        dtype=torch.float,
        device=torch.device("cpu"),
        hidden_sizes=(100,),
        activation=nn.ReLU(),
        max_num_actions=10,
        log_epsilon=-20.0,
        *args,
        **kwargs
    ):
        super().__init__(env, gamma, optim_kwargs, history_length, dtype, device)

        self.max_num_actions = max_num_actions
        self.log_epsilon = log_epsilon

        self.preprocessor = SechsNimmtStateNormalization(action=True)
        self.actor_critic = MultiHeadedMLP(
            1 + self.state_length, hidden_sizes=hidden_sizes, head_sizes=(1, 1), activation=activation, head_activations=(None, None),
        )
        self.softmax = nn.Softmax(dim=0)

    def forward(self, state, legal_actions, **kwargs):
        batch_states = self._batch_state(state, legal_actions)
        log_probs, qs = self._evaluate(batch_states)
        action_id = self._act(log_probs, legal_actions)

        return (
            legal_actions[action_id],
            {"action_id": action_id, "log_probs": log_probs, "log_prob": log_probs[action_id], "values": qs, "value": qs[action_id]},
        )

    def evaluate(self, states, legal_actions_list):
        all_qs = []
        all_log_probs = []

        for state, legal_actions in zip(states, legal_actions_list):  # TODO batch
            batch_states = self._batch_state(state, legal_actions)
            log_probs, qs = self._evaluate(batch_states)
            all_qs.append(qs.flatten().unsqueeze(0))
            all_log_probs.append(log_probs.unsqueeze(0))

        all_qs = torch.cat(all_qs, dim=0)
        all_log_probs = torch.cat(all_log_probs, dim=0)

        return all_log_probs, all_qs

    def learn(self, state, reward, action, done, next_state, next_reward, episode_end, num_episode, legal_actions, *args, **kwargs):
        raise NotImplementedError

    def _batch_state(self, state, legal_actions):
        batch_states = []
        for action in legal_actions:
            action_ = torch.tensor([action]).to(self.device, self.dtype)
            batch_states.append(torch.cat((action_, state), dim=0).unsqueeze(0))
        batch_states = torch.cat(batch_states, dim=0)
        return batch_states

    def _evaluate(self, batch_states, pad=True):
        batch_states = self.preprocessor(batch_states)
        (probs, qs) = self.actor_critic(batch_states)
        probs = self.softmax(probs).flatten()
        qs = qs.flatten()
        log_probs = torch.log(probs)

        if pad:
            qs = self._pad(qs, value=0.0)
            log_probs = self._pad(log_probs)

        return log_probs, qs

    def _act(self, log_probs, legal_actions):
        cat = Categorical(torch.exp(torch.clamp(log_probs, -20)))
        action_id = len(legal_actions)
        while action_id >= len(legal_actions):
            action_id = cat.sample()
        return action_id

    def _pad(self, inputs, value=None):
        return torch.nn.functional.pad(
            inputs, (0, self.max_num_actions - inputs.size()[-1]), mode="constant", value=self.log_epsilon if value is None else value
        )

    def _gradient_step(self, loss):
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()


class BatchedACERAgent(BatchedActionValueActorCriticAgent):
    """ Based on https://github.com/seungeunrho/minimalRL/blob/master/acer.py """

    def __init__(self, *args, rollout_len=10, minibatch=5, truncate=1.0, warmup=100, r_factor=0.1, actor_weight=1.0, critic_weight=1.0, **kwargs):
        self.truncate = truncate
        self.warmup = warmup
        self.batchsize = minibatch
        self.rollout_len = rollout_len
        self.r_factor = r_factor
        self.actor_weight = actor_weight
        self.critic_weight = critic_weight

        super().__init__(*args, **kwargs)

    def _init_replay_buffer(self, history_length):
        self.history = SequentialHistory(max_length=history_length, dtype=self.dtype, device=self.device)

    def learn(self, state, reward, action, done, next_state, next_reward, episode_end, num_episode, legal_actions, *args, **kwargs):
        self.history.store(
            state=state,
            legal_actions=legal_actions,
            log_probs=kwargs["log_probs"],
            action_id=kwargs["action_id"],
            next_reward=next_reward * self.r_factor,
            done=done,
        )

        if self.history.current_sequence_length() >= self.rollout_len or done or episode_end:
            self.history.flush()

            if len(self.history) > max(self.warmup, self.batchsize):
                self._train(on_policy=True)
                self._train(on_policy=False)

    def _train(self, on_policy=True):
        # Rollout
        action_ids, done, is_first, legal_actions_list, log_probs_then, rewards, states = self._rollout(on_policy)

        log_probs_now, q = self.evaluate(states, legal_actions_list)
        q_a = q.gather(1, action_ids)
        log_prob_now_a = log_probs_now.gather(1, action_ids)
        v = (q * torch.exp(log_probs_now)).sum(1).unsqueeze(1).detach()

        rho = torch.exp(log_probs_now - log_probs_then).detach()
        rho_a = rho.gather(1, action_ids)
        rho_bar = rho_a.clamp(max=self.truncate)
        correction_coeff = (1.0 - self.truncate / rho).clamp(min=0.0)

        q_ret = self._compute_return(done, is_first, q_a, rewards, rho_bar, v)

        actor_loss = -rho_bar * log_prob_now_a * (q_ret - v)
        actor_loss = actor_loss.mean()
        correction_loss = -correction_coeff * torch.exp(log_probs_then.detach()) * log_probs_now * (q.detach() - v)  # bias correction term
        correction_loss = correction_loss.sum(1).mean()
        critic_loss = self.critic_weight * torch.nn.SmoothL1Loss()(q_a, q_ret)

        self._gradient_step(actor_loss + correction_loss + critic_loss)

        return actor_loss.item(), correction_loss.item(), critic_loss.item()

    def _rollout(self, on_policy):
        if on_policy:
            rollout = self.history.rollout(n=1)
        else:
            _, _, rollout = self.history.sample(self.batchsize)

        states = torch.stack(list(iter_flatten(rollout["state"])))
        legal_actions_list = list(iter_flatten(rollout["legal_actions"], max_depth=1))
        action_ids = torch.tensor(list(iter_flatten(rollout["action_id"])), dtype=torch.long).unsqueeze(1)
        rewards = np.array(list(iter_flatten(rollout["next_reward"])))
        log_probs_then = torch.stack(list(iter_flatten(rollout["log_probs"])))
        done = np.array(list(iter_flatten(rollout["done"])), dtype=np.bool)
        is_first = np.array(list(iter_flatten(rollout["first"])), dtype=np.bool)

        return action_ids, done, is_first, legal_actions_list, log_probs_then, rewards, states

    def _compute_return(self, done, is_first, q_a, rewards, rho_bar, v):
        q_ret = v[-1] * (1.0 - done[-1])
        q_ret_lst = []
        for i in reversed(range(len(rewards))):
            q_ret = rewards[i] + self.gamma * q_ret
            q_ret_lst.append(q_ret.item())
            q_ret = rho_bar[i] * (q_ret - q_a[i]) + v[i]

            if is_first[i] and i != 0:
                q_ret = v[i - 1] * (1.0 - done[i - 1])  # When a new sequence begins, q_ret is initialized
        q_ret_lst.reverse()
        q_ret = torch.tensor(q_ret_lst, dtype=torch.float).unsqueeze(1)
        return q_ret
