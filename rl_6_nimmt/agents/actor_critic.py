import torch
from torch import nn
from torch.distributions import Categorical
import numpy as np

from .base import Agent
from ..utils.nets import MultiHeadedMLP
from ..utils.various import compute_discounted_returns, iter_flatten
from ..utils.replay_buffer import SequentialHistory
from ..utils.preprocessing import SechsNimmtStateNormalization


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
        *args,
        **kwargs
    ):
        super().__init__(env, gamma, optim_kwargs, history_length, dtype, device)

        # NN that calculates the policy (actor) and estimates Q (critic)
        self.preprocessor = SechsNimmtStateNormalization(action=True)
        self.actor_critic = MultiHeadedMLP(
            1 + self.state_length,
            hidden_sizes=hidden_sizes,
            head_sizes=(1, 1),
            activation=activation,
            head_activations=(None, None),
        )
        self.softmax = nn.Softmax(dim=0)

    def forward(self, state, legal_actions, **kwargs):
        # Let the actor pick action probabilities and the critic guess their Q-values
        probs, values = self.actor_critic(state)
        log_probs = torch.log(probs)

        # Sample action from these probabilities
        cat = Categorical(probs)
        action = cat.sample()
        entropy = cat.entropy()
        log_prob = log_probs[action]
        value = values[action]

        return action.item(), {"log_probs": log_probs, "log_prob": log_prob, "values": values, "value": value, "entropy": entropy}

    def evaluate_states(self, states):
        # Let the actor pick action probabilities and the critic guess the expected reward V(s_t)
        probs, values = self.actor_critic(states)
        log_probs = torch.log(probs)
        entropies = Categorical(probs).entropy()

        return log_probs, values, entropies

    def evaluate_action(self, state, action):
        """
        Given an environment state and an action, return the log likelihood and an estimate of the state value.

        Parameters
        ----------
        state : Tensor
            Observed state s_t

        action : int
            Fixed action a_t

        Returns
        -------

        log_prob : torch.Tensor
            Log probability of chosen action under the policy pi(a_t|s_t).

        value : torch.Tensor
            Estimated state-action value q(s_t, a_t).

        entropy : torch.Tensor
            Entropy sum_i p(a_i) log p(a_i).

        """

        # Let the actor pick action probabilities and the critic guess the expected reward V(s_t)
        probs, values = self.actor_critic(state)

        # Sample action from these probabilities
        cat = Categorical(probs)
        log_prob = cat.log_prob(action)
        entropy = cat.entropy()
        value = values[:, action]

        return log_prob, value, entropy

    def learn(self, state, reward, action, done, next_state, next_reward, episode_end, gamma, *args, **kwargs):
        """
        Is called at the end of each step, gives the agent the chance to a) update the replay buffer and b) learn its weights.
        """
        raise NotImplementedError

    def _gradient_step(self, loss):
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()


class BatchedACERAgent(BatchedActionValueActorCriticAgent):
    """ Based on https://github.com/seungeunrho/minimalRL/blob/master/acer.py """

    def __init__(self, *args, rollout_len=10, minibatch=4, truncate=1.0, warmup=500, r_factor=0.01, actor_weight=1.0, critic_weight=1.0, **kwargs):
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

    def learn(self, state, reward, action, done, next_state, next_reward, episode_end, num_episode, *args, **kwargs):
        """
        Is called at the end of each step, gives the agent the chance to a) update the replay buffer and b) learn its weights.
        """

        # Memorize step
        self.history.store(state=state, log_probs=kwargs["log_probs"], action=action, next_reward=next_reward * self.r_factor, done=done)

        losses = np.zeros(6)

        # Every self.rollout_len transitions, we train
        if self.history.current_sequence_length() >= self.rollout_len or done or episode_end:
            self.history.flush()

            if len(self.history) > self.warmup:
                losses[:3] = self._train(on_policy=True)
                losses[3:] = self._train(on_policy=False)

        return losses

    def _train(self, on_policy=True):
        # Rollout
        if on_policy:
            rollout = self.history.rollout(n=1)
        else:
            _, _, rollout = self.history.sample(self.batchsize)

        states = torch.stack(list(iter_flatten(rollout["state"])))
        actions = torch.tensor(list(iter_flatten(rollout["action"])), dtype=torch.long).unsqueeze(1)
        rewards = np.array(list(iter_flatten(rollout["next_reward"])))
        log_probs_then = torch.stack(list(iter_flatten(rollout["log_probs"])))
        done = np.array(list(iter_flatten(rollout["done"])), dtype=np.bool)
        is_first = np.array(list(iter_flatten(rollout["first"])), dtype=np.bool)

        log_probs_now, q, _ = self.evaluate_states(states)
        q_a = q.gather(1, actions)
        log_prob_now_a = log_probs_now.gather(1, actions)
        v = (q * torch.exp(log_probs_now)).sum(1).unsqueeze(1).detach()

        rho = torch.exp(log_probs_now - log_probs_then).detach()
        rho_a = rho.gather(1, actions)
        rho_bar = rho_a.clamp(max=self.truncate)
        correction_coeff = (1.0 - self.truncate / rho).clamp(min=0.0)

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

        actor_loss1 = -rho_bar * log_prob_now_a * (q_ret - v)
        actor_loss2 = -correction_coeff * torch.exp(log_probs_then.detach()) * log_probs_now * (q.detach() - v)  # bias correction term
        actor_loss2 = actor_loss2.sum(1)
        critic_loss = self.critic_weight * torch.nn.SmoothL1Loss()(q_a, q_ret)

        self._gradient_step(actor_loss1 + actor_loss2 + critic_loss)

        return actor_loss1.mean().item(), actor_loss2.mean().item(), critic_loss.mean().item()



# class StateValueActorCriticAgent(Agent):
#     """
#     Simple actor-critic agent for discrete action spaces.
#
#     The actor implements the policy pi(a|s). The critic estimates the state value v(s). The base A-C class does not yet implement any training algorithm.
#     """
#
#     def __init__(
#         self,
#         env=None,
#         gamma=0.99,
#         optim_kwargs=None,
#         history_length=None,
#         dtype=torch.float,
#         device=torch.device("cpu"),
#         hidden_sizes=(100,),
#         activation=nn.ReLU(),
#         *args,
#         **kwargs
#     ):
#         super().__init__(env, gamma, optim_kwargs, history_length, dtype, device)
#
#         # NN that calculates the policy (actor) and estimates Q (critic)
#         self.actor_critic = MultiHeadedMLP(
#             self.state_length, hidden_sizes=hidden_sizes, head_sizes=(self.num_actions, 1), activation=activation, head_activations=(nn.Softmax(dim=-1), None)
#         )
#
#     def forward(self, state, legal_actions, **kwargs):
#         """
#         Given an environment state, pick the next action and return. Additional outputs are the estimated state value, the log likelihood of the policy, and its entropy.
#
#         Parameters
#         ----------
#         state : Tensor
#             Observed state s_t
#
#         Returns
#         -------
#         action : int
#             Chosen action a_t
#
#         agent_info : dict
#             Additional agent output, in this case the log likelihood and the entropy.
#
#         """
#
#         # Let the actor pick action probabilities and the critic guess the expected reward V(s_t)
#         probs, value = self.actor_critic(state)
#
#         # Sample action from these probabilities
#         cat = Categorical(probs)
#         action = cat.sample()
#         log_prob = cat.log_prob(action)
#         entropy = cat.entropy()
#
#         return action.item(), {"log_prob": log_prob, "value": value, "entropy": entropy}
#
#     def evaluate(self, state, action):
#         """
#         Given an environment state and an action, return the log likelihood and an estimate of the state value.
#
#         Parameters
#         ----------
#         state : Tensor
#             Observed state s_t
#
#         action : int
#             Fixed action a_t
#
#         Returns
#         -------
#
#         log_prob : torch.Tensor
#             Log probability of chosen action under the policy pi(a_t|s_t).
#
#         value : torch.Tensor
#             Estimated state value v(s_t).
#
#         entropy : torch.Tensor
#             Entropy sum_i p(a_i) log p(a_i).
#
#         """
#
#         # Let the actor pick action probabilities and the critic guess the expected reward V(s_t)
#         probs, value = self.actor_critic(state)
#
#         # Sample action from these probabilities
#         cat = Categorical(probs)
#         log_prob = cat.log_prob(action)
#         entropy = cat.entropy()
#
#         return log_prob, value, entropy
#
#     def learn(self, state, reward, action, done, next_state, next_reward, episode_end, num_episode, *args):
#         """
#         Is called at the end of each step, gives the agent the chance to a) update the replay buffer and b) learn its weights.
#         """
#         raise NotImplementedError
#
#     def _gradient_step(self, loss):
#         self.optimizer.zero_grad()
#         loss.backward()
#         self.optimizer.step()

# class MCActorCriticAgent(StateValueActorCriticAgent):
#     """
#     On-policy REINFORCE training with a learned critic baseline / actor-critic training based on MC returns.
#
#     The actor is trained at the end of episodes based on the policy gradient theorem using the critic output as baseline.
#     The critic learns v(s) and is trained at the end of episodes based on MC returns.
#     """
#
#     def __init__(self, *args, r_factor=1.0, actor_weight=1.0, critic_weight=1.0, entropy_weight=0.0, **kwargs):
#         self.entropy_weight = entropy_weight
#         self.r_factor = r_factor
#         self.actor_weight = actor_weight
#         self.critic_weight = critic_weight
#         super().__init__(*args, **kwargs)
#
#     def learn(self, state, reward, action, done, next_state, next_reward, episode_end, num_episode, *args, **kwargs):
#         """
#         Is called at the end of each step, gives the agent the chance to a) update the replay buffer and b) learn its weights.
#         """
#
#         # Memorize step
#         self.history.store(log_prob=kwargs["log_prob"], state=state, value=kwargs["value"], reward=reward * self.r_factor, entropy=kwargs["entropy"])
#
#         # No further steps after each step
#         if not episode_end or not self.training:
#             return np.zeros(3)
#
#         losses = self._train()
#
#         # Reset memory for next episode
#         self.history.clear()
#
#         return losses
#
#     def _train(self):
#         # Roll out last episode
#         rollout = self.history.rollout()
#         n = len(self.history)
#         log_probs = torch.stack(rollout["log_prob"], dim=0)
#         entropies = torch.stack(rollout["entropy"], dim=0)
#         returns = torch.tensor(compute_discounted_returns(rollout["reward"], self.gamma, dtype=self.dtype, device=self.device)).squeeze()
#         values = torch.stack(rollout["value"], dim=0).squeeze()
#         advantages = returns - values
#         returns_norm = (returns - returns.mean()) / (returns.std())
#
#         # Compute loss
#         discounts = torch.exp(np.log(self.gamma) * torch.linspace(0, n - 1, n)).to(self.device, self.dtype)
#         actor_loss = -torch.sum(discounts * advantages.detach() * log_probs)
#         critic_loss = torch.nn.MSELoss(reduction="sum")(values, returns_norm)
#
#         # Entropy regularization to incentivize exploration
#         if entropies is not None:
#             entropy_loss = -torch.sum(entropies)
#         else:
#             entropy_loss = torch.tensor(0.0)
#
#         # Gradient update
#         self._gradient_step(self.actor_weight * actor_loss + self.critic_weight * critic_loss + self.entropy_weight * entropy_loss)
#
#         return np.array([actor_loss.item(), critic_loss.item(), entropy_loss.item()])
#
#
# class NStepActorCriticAgent(StateValueActorCriticAgent):
#     """
#     On-policy n-step actor-critic training.
#
#     Actor and critic are trained after every step based on TD(n) differences. The critic learns v(s).
#     """
#
#     def __init__(self, *args, n_steps=3, warmup=500, r_factor=0.01, actor_weight=1.0, critic_weight=1.0, entropy_weight=0.0, **kwargs):
#         self.n = n_steps
#         self.warmup = warmup
#         self.r_factor = r_factor
#         self.actor_weight = actor_weight
#         self.critic_weight = critic_weight
#         self.entropy_weight = entropy_weight
#         super().__init__(*args, **kwargs)
#
#     def learn(self, state, reward, action, done, next_state, next_reward, episode_end, num_episode, *args, **kwargs):
#         """
#         Is called at the end of each step, gives the agent the chance to a) update the replay buffer and b) learn its weights.
#         """
#
#         # Memorize step
#         self.history.store(state=state, action=action, next_reward=next_reward * self.r_factor, next_state=next_state, done=done)
#
#         # No updates
#         if len(self.history) < self.n and not done:
#             return np.zeros(3)
#
#         # Train
#         losses = self._train()
#
#         # Reset memory
#         self.history.clear()
#
#         return losses
#
#     def _train(self):
#         # Backup
#         rollout = self.history.rollout()
#         done = len(self.history) > 0 and bool(rollout["done"][-1])
#
#         # Episode completed before n steps
#         t = len(self.history) - self.n
#         n = self.n
#         if t < 0:
#             t = 0
#             n = len(self.history)  # TODO
#
#         # Load starting point of our n-step return
#         rewards = torch.tensor(rollout["next_reward"][-n:])
#         state = rollout["state"][-n]
#         action = torch.tensor(rollout["action"][-n])
#         next_state = rollout["next_state"][-1]
#
#         # Evaluate original and horizon state and compute n-step return
#         log_prob, value, entropy = self.evaluate(state, action)
#         if done:
#             next_value = 0.0
#         else:
#             _, next_value, _ = self.evaluate(next_state, action)
#         nstep_return = compute_discounted_returns(rewards, self.gamma, self.dtype, self.device)[0] + self.gamma ** n * next_value
#         advantage = nstep_return - value
#
#         # Loss
#         actor_loss = -np.exp(np.log(self.gamma) * t) * advantage.detach().squeeze() * log_prob
#         critic_loss = torch.nn.MSELoss(reduction="sum")(value.squeeze(), nstep_return.detach().squeeze())
#
#         # Entropy regularization to incentivize exploration
#         if entropy is not None:
#             entropy_loss = -entropy
#         else:
#             entropy_loss = torch.tensor(0.0)
#
#         # Gradient update
#         self._gradient_step(self.actor_weight * actor_loss + self.critic_weight * critic_loss + self.entropy_weight * entropy_loss)
#
#         return np.array([actor_loss.item(), critic_loss.item(), entropy_loss.item()])