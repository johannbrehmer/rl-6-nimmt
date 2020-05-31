"""
Different kind of DQNs
vanilla DQN -> Q(s,a) = R + gamma * max[ Q(s_next, a_next')]
Double DQN ->  Q(s,a) = R + gamma * Q'(s_next, argmax Q(s_next, a_next'))
"""

import torch
from torch import nn

# from torch.distributions import Categorical

from .base import Agent

# from collections import namedtuple, deque
import random
import numpy as np
import logging

import math

from ..utils.nets import MultiHeadedMLP, DuellingDQNNet, CNN, NoisyLinear, NoisyFactorizedLinear
from ..utils.replay_buffer import History, PriorityReplayBuffer
from ..utils.various import plot_grad_flow

logger = logging.getLogger("dqn_agent")
# constants
STATE = "state"
ACTION = "action"
REWARD = "reward"
NEXT_STATE = "next_state"
DONE = "done"


def eps_func_decay(episode):
    rel_val = episode
    decay_rate = 0.0025
    eps = 1 * math.exp(-rel_val * decay_rate)
    eps = max(eps, 0.05)
    return eps


class DQNVanilla(Agent):
    """ Deep Q-learning """

    def __init__(self, *args, hidden_sizes=(64,), activation=nn.ReLU(), n_steps=1, eps_func=None, minibatch=64, summary_writer=None, **kwargs):

        super().__init__(*args, **kwargs)

        self.summary_writer = summary_writer
        self.n_steps = n_steps  # for easier n_step inheritance
        self.minibatch = minibatch

        self.eps_func = eps_func
        if eps_func is None:
            self.eps_func = eps_func_decay

        self.step = 0
        self.eps = 0  # eps = 0 means always best learned action is chosen, eps = 1, means always random action is chosen

        self._setup_networks(hidden_sizes, activation)

        # self._init_optimizer()

    def _setup_networks(self, hidden_sizes, activation):
        self.dqn_net_local = MultiHeadedMLP(
            self.state_length, hidden_sizes=hidden_sizes, head_sizes=(self.num_actions,), activation=activation, head_activations=(None,)
        )

    def train(self, mode=True):
        super().train(mode=mode)
        self.dqn_net_local.train()

        self.eps = self.eps_func(0)

    def _init_replay_buffer(self, history_length):
        self.history = History(max_length=history_length, dtype=self.dtype, device=self.device)

    def _min_history_length(self):
        return self.minibatch

    def _store(self, **kwargs):
        self.history.store(**kwargs)

    def _finish_episode(self):
        pass

    def learn(self, state, reward, action, done, next_state, next_reward, episode_end, num_episode, legal_actions, *args, **kwargs):

        self.step += 1

        loss = 0
        self.eps = self.eps_func(num_episode)
        # if self.summary_writer is not None and done:
        #     for i, k in enumerate(state):
        #         self.summary_writer.add_scalar(f"debug/state_{i}_when_done", k, num_episode)

        if self.summary_writer is not None and episode_end:
            self.summary_writer.add_scalar("debug/eps", self.eps, num_episode)
        self._store(state=state, reward=reward, action=action, next_state=next_state, done=done)

        if len(self.history) > self._min_history_length():

            loss = self._learn(num_episode, episode_end)

        if done:
            self._finish_episode()

        return np.array([loss])

    def _learn(self, num_episode, episode_end):
        """Update value parameters using given batch of experience tuples.
        Params
        =======
            experiences (Tuple[torch.Variable]): tuple of (s, a, r, s', done) tuples
            gamma (float): discount factor
        """
        memory_idx, importance_sampling_weights, experiences = self.history.sample(n=self.minibatch)
        # states, actions, rewards, next_state, dones = experiences
        experiences = self._prep_minibatch(experiences)  # non_final_next_states are only needed for n-step variants

        states = experiences[STATE]
        actions = experiences[ACTION]
        rewards = experiences[REWARD]
        next_states = experiences[NEXT_STATE]
        dones = experiences[DONE]

        q = self.dqn_net_local(states)[0]

        q_eval = q.gather(1, actions.view(-1, 1))  # shape (batch, 1)
        q_target = self._calc_reward(next_states, rewards, dones)
        q_eval = torch.squeeze(q_eval)
        q_target = torch.squeeze(q_target)

        if self.summary_writer is not None and episode_end and num_episode % 10 == 0:
            self.summary_writer.add_scalar("debug/bellman_target", q_target.max(), num_episode)

        # ------------------- update target network ------------------- #
        self._update_memory(q_eval.detach(), q_target.detach(), memory_idx)
        loss = self._optimize_loss(q_eval, q_target, importance_sampling_weights)

        return loss.item()

    def _prep_minibatch(self, experiences):
        for key, item in experiences.items():
            if isinstance(item[0], torch.Tensor):
                experiences[key] = torch.stack(item, dim=0).to(self.device)
            else:
                experiences[key] = torch.from_numpy(np.array(item))

            # if isinstance(item, list) and isinstance(item[0], (float)):
            #     experiences[key] = torch.Tensor(item).to(self.device)
            # elif isinstance(item, list) and isinstance(item[0], (int, bool, np.int)):
            #     experiences[key] = torch.from_numpy(item).to(self.device, dtype = torch.int)
            # # elif isinstance(item, list) and isinstance(item[0], (float, int, np.number)):
            # #     experiences[key] = torch.from_numpy(np.array(item)).to(self.device)
            # elif isinstance(item, np.ndarray):
            #     experiences[key] = torch.from_numpy(item).to(self.device)
            # elif isinstance(item, torch.Tensor):
            #     experiences[key] = item.squeeze().to(self.device)
            # elif isinstance(item[0], torch.Tensor):
            #     experiences[key] = torch.stack(item, dim=0).to(self.device)
            # elif key == "action":
            #     experiences[key] = torch.from_numpy(item).long().to(self.device)
            # elif isinstance(item[0], torch.Tensor):
            #     experiences[key] = torch.cat(item)
            # else:
            #     raise NotImplementedError(f"Unkown type {type(item[0])}")
        return experiences

    def _optimize_loss(self, q_eval, q_target, weights):
        criterion = torch.nn.MSELoss()
        loss = criterion(q_eval, q_target).to(self.device)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        return loss

    def _update_memory(self, target_old, target, memory_idx):
        pass

    def _calc_reward(self, next_states, rewards, done):
        """
        Reward function for vanilla DQN
        """
        with torch.no_grad():
            done_mask = done == 1
            q_predict = self.dqn_net_local(next_states)[0].detach()
            next_max_q_values, indices = torch.max(q_predict, axis=-1)
            next_max_q_values = next_max_q_values * ~done_mask

            q_targets = rewards + self.gamma ** self.n_steps * next_max_q_values

        return q_targets

    def _action_selection(self, scores, **kwargs):
        """epsilon greedy slection"""

        legal_actions = kwargs.get('legal_actions', None)

        if legal_actions:
            illegal_actions = list(set(np.arange(self.num_actions)) - set(legal_actions))
            scores[illegal_actions] = -1e8

        if random.random() > self.eps:
            value = np.max(scores.cpu().data.numpy())
            action_id = np.argmax(scores.cpu().data.numpy())
            #action = actions[action_id]
            logger.debug(f"action: {action_id}. eps: {self.eps}, value: {value}")
        else:
            action_id = random.choice(np.arange(self.num_actions))
            if legal_actions:
                action_id = random.choice(legal_actions)
            value = -1
            logger.debug(f"random action: {action_id}. eps: {self.eps}")

        return action_id, {"value": value, "eps": self.eps}

    def forward(self, state, **kwargs):
        """
        Given an environment state, pick the next action and return it together with the log likelihood and an estimate of the state value.


        """
        # state = torch.from_numpy(state).unsqueeze(0)
        self.dqn_net_local.eval()
        with torch.no_grad():
            scores = self.dqn_net_local(state)[0]
        self.dqn_net_local.train()
        return self._action_selection(scores, **kwargs)
        # Epsilon -greedy action selection


class Noisy_DQN(DQNVanilla):
    # from https://github.com/Shmuma/ptan/blob/master/samples/rainbow/lib/dqn_model.py
    def __init__(self, *args, noisy_init_sigma=0.5, **kwargs):
        self.noisy_init_sigma = noisy_init_sigma
        super().__init__(*args, **kwargs)

    def _setup_networks(self, hidden_sizes, activation):
        self.dqn_net_local = MultiHeadedMLP(
            self.state_length,
            hidden_sizes=hidden_sizes,
            head_sizes=(self.num_actions,),
            activation=activation,
            head_activations=(None,),
            linear=NoisyFactorizedLinear,
            init_sigma=self.noisy_init_sigma,
        )

    def _action_selection(self, scores, **kwargs):
        """argmax selection"""
        legal_actions = kwargs.get('legal_actions', None)
        actions = np.arange(self.num_actions)
        if legal_actions:
            scores = scores[legal_actions]
            actions = actions[legal_actions]
        value = np.max(scores.cpu().data.numpy())
        action_id = np.argmax(scores.cpu().data.numpy())
        action = actions[action_id]
        return action, {"value": value}


class DQN_NStep_Agent(DQNVanilla):
    # modified from https://github.com/qfettes/DeepRL-Tutorials/blob/master/02.NStep_DQN.ipynb
    def __init__(self, *args, **kwargs):
        self.n_step_buffer = []
        super().__init__(*args, **kwargs)

    def _store(self, **kwargs):
        self.n_step_buffer.append(kwargs)
        if len(self.n_step_buffer) < self.n_steps:
            return
        R = sum([self.n_step_buffer[i]["reward"] * (self.gamma ** i) for i in range(self.n_steps)])
        n_step_experience = self.n_step_buffer.pop(0)
        n_step_experience[REWARD] = R
        n_step_experience[NEXT_STATE] = kwargs[NEXT_STATE]
        self.history.store(**n_step_experience)

    # def _prep_minibatch(self, experiences):
    #     non_final_next_states = tuple(map(lambda s: s is not None, experiences[NEXT_STATE]))
    #     assert np.sum(non_final_next_states) > 0
    #     # for key, val in experiences.items():
    #     #     experiences[key] = [x for i,x in enumerate(val) if is_non_final[i]] # hm looks clumsy, any better idea?
    #     experiences = super()._prep_minibatch(experiences)
    #     return experiences

    def _finish_episode(self):
        if len(self.n_step_buffer) == 0:
            return
        last_experience = self.n_step_buffer[-1]
        while len(self.n_step_buffer) > 0:
            R = sum([self.n_step_buffer[i][REWARD] * (self.gamma ** i) for i in range(len(self.n_step_buffer))])
            n_step_experience = self.n_step_buffer.pop(0)
            n_step_experience[REWARD] = R
            n_step_experience[NEXT_STATE] = last_experience[NEXT_STATE]
            # check, technically done references to the current state, not the next state. But in calculations it makes
            # no difference, since it is only a mask to remove "invalid states"
            # TODO check if needed at all, if we are not using memories at the end of an episode, why bother saving them? Only for q eval of current state?!
            n_step_experience[DONE] = True
            self.history.store(**n_step_experience)


class DDQNAgent(DQNVanilla):
    def __init__(self, *args, retrain_interval=4, tau=1e-2, **kwargs):

        self.tau = tau
        self.retrain_interval = retrain_interval
        super().__init__(*args, **kwargs)

    def _setup_networks(self, hidden_sizes, activation):
        self.dqn_net_local = MultiHeadedMLP(
            self.state_length, hidden_sizes=hidden_sizes, head_sizes=(self.num_actions,), activation=activation, head_activations=(None,)
        )

        self.dqn_net_target = MultiHeadedMLP(
            self.state_length, hidden_sizes=hidden_sizes, head_sizes=(self.num_actions,), activation=activation, head_activations=(None,)
        )
        self.dqn_net_target.eval()
        # have same params for local and target net
        self.soft_update(self.dqn_net_local, self.dqn_net_target, 1)

    def _calc_reward(self, next_states, rewards, done):
        """
        Reward function for double DQN
        """
        done_mask = done == 1
        done_mask = done_mask.squeeze()
        with torch.no_grad():
            next_q_values_target = self.dqn_net_target(next_states)[0].detach()
            next_q_values_local = self.dqn_net_local(next_states)[0].detach()
            idx = torch.argmax(next_q_values_local, axis=-1).squeeze()  # get action from local net

            q_term = next_q_values_target[torch.arange(next_q_values_target.shape[0]), idx]  # get Q value associate with this action from target net
            q_term = q_term * ~done_mask
            q_targets = rewards + self.gamma ** self.n_steps * q_term
        if (self.step % self.retrain_interval) == 0:
            self.soft_update(self.dqn_net_local, self.dqn_net_target, self.tau)

        return q_targets

    def soft_update(self, local_model, target_model, tau):
        """Soft update model parameters.
        θ_target = τ*θ_local + (1 - τ)*θ_target
        Params
        =======
            local model (PyTorch model): weights will be copied from
            target model (PyTorch model): weights will be copied to
            tau (float): interpolation parameter
        """
        logger.debug("Updating target net")
        for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
            target_param.data.copy_(tau * local_param.data + (1 - tau) * target_param.data)


class DQN_PRBAgent(DQNVanilla):
    def _init_replay_buffer(self, history_length):
        self.history = PriorityReplayBuffer(history_length, device=self.device, dtype=self.dtype)

    def _update_memory(self, target_old, target, memory_idx):
        abs_errs = torch.abs(target_old - target).view(memory_idx.shape)
        self.history.batch_update(memory_idx, abs_errs)

    def _optimize_loss(self, q_eval, q_target, weights):
        weights = torch.from_numpy(weights).to(self.device, self.dtype)
        loss = weights * (q_eval - q_target) ** 2
        # loss =  (q_eval - q_target) ** 2
        loss = loss.mean()
        self.optimizer.zero_grad()
        loss.backward()
        # if self.step % 200 == 0:

        # for tag, param in self.dqn_net_local.modules().iteritems():#self.named_parameters():
        #     if param.grad is not None:
        #         self.summary_writer.add_histogram(tag, param.grad.data.cpu().numpy(), self.step)

        # plot_grad_flow(self.named_parameters())
        self.optimizer.step()
        return loss

    # def _min_history_length(self):
    #     return self.history.tree.capacity


class DuellingDQNAgent(DQNVanilla):
    """
    network is split into output for V and A, which are related
    by Q = V + A, which is either represented
    as Q = V +(A - max_a(A)) or
    as Q = V +(A - mean(A)) to avoid the "unidentifiablity", meaning the recovery of V and A unambigously from Q
    """

    def _setup_networks(self, hidden_sizes, activation):
        self.dqn_net_local = DuellingDQNNet(self.state_length, hidden_sizes=hidden_sizes, out_size=self.num_actions, activation=activation)


class DuellingDDQNAgent(DDQNAgent):
    def _setup_networks(self, hidden_sizes, activation):
        self.dqn_net_local = DuellingDQNNet(self.state_length, hidden_sizes=hidden_sizes, out_size=self.num_actions, activation=activation)
        self.dqn_net_target = DuellingDQNNet(self.state_length, hidden_sizes=hidden_sizes, out_size=self.num_actions, activation=activation)
        self.soft_update(self.dqn_net_local, self.dqn_net_target, 1)  # tau = 1 corresponds to a hard update


class Noisy_D3QN(DuellingDDQNAgent, Noisy_DQN):
    # from https://github.com/Shmuma/ptan/blob/master/samples/rainbow/lib/dqn_model.py

    def _setup_networks(self, hidden_sizes, activation):
        self.dqn_net_local = DuellingDQNNet(
            self.state_length,
            hidden_sizes=hidden_sizes,
            out_size=self.num_actions,
            activation=activation,
            linear=NoisyFactorizedLinear,
            init_sigma=self.noisy_init_sigma,
        )
        self.dqn_net_target = DuellingDQNNet(
            self.state_length,
            hidden_sizes=hidden_sizes,
            out_size=self.num_actions,
            activation=activation,
            linear=NoisyFactorizedLinear,
            init_sigma=self.noisy_init_sigma,
        )
        self.soft_update(self.dqn_net_local, self.dqn_net_target, 1)  # tau = 1 corresponds to a hard update


class DDQN_PRBAgent(DQN_PRBAgent, DDQNAgent):
    pass


class DuellingDDQN_PRBAgent(DQN_PRBAgent, DuellingDDQNAgent):
    pass


class D3QN_PRB_NStep(DQN_NStep_Agent, DuellingDDQN_PRBAgent):
    pass


class Noisy_D3QN_PRB_NStep(Noisy_D3QN, D3QN_PRB_NStep):
    pass
