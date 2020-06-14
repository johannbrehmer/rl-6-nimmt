import math
import torch
from torch import nn
from torch.distributions import Categorical
import numpy as np
import logging

from ..env import SechsNimmtEnv
from .base import Agent
from ..utils.nets import MultiHeadedMLP
from ..utils.preprocessing import SechsNimmtStateNormalization

logger = logging.getLogger(__name__)



class BaseMCAgent(Agent):
    def __init__(
        self,
        handsize=10,
        num_rows=4,
        num_cards=104,
        threshold=6,
        mc_per_card=10,
        mc_max=100,
        include_summaries=True,
        *args,
        **kwargs
    ):
        super().__init__(*args, **kwargs)

        self.num_players = None  # will later be inferred from states
        self.handsize = handsize
        self.num_rows = num_rows
        self.num_cards = num_cards
        self.threshold = threshold
        self.mc_per_card = mc_per_card
        self.mc_max = mc_max
        self.include_summaries = include_summaries

        self.available_cards = []

    def forward(self, state, legal_actions, *args, **kwargs):
        n = len(legal_actions)

        # Card memory
        if n == self.handsize:
            self._initialize_game(state)
        self._memorize_cards(state, legal_actions)

        # Pick action through a kind of MCTS
        if n == 1:
            action, info = legal_actions[0], {"log_prob": torch.tensor(0.).to(self.device, self.dtype)}
        else:
            action, info = self._mcts(legal_actions, state)

        return action, info

    def learn(self, state, reward, action, done, next_state, next_reward, episode_end, num_episode, legal_actions, *args, **kwargs):
        raise NotImplementedError

    def _initialize_game(self, state):
        self.available_cards = list(range(self.num_cards))
        self.num_players = self._num_players_from_state(state)

    def _memorize_cards(self, state, legal_actions):
        for card in legal_actions + self._board_from_state(state, flatten=True):
            if card < 0:
                continue
            try:
                self.available_cards.remove(card)
            except:
                pass

    def _board_from_state(self, state, flatten=True):
        board_array = state[-self.num_rows * self.threshold:].reshape((self.num_rows, self.threshold))

        board = []
        for row in board_array:
            if flatten:
                board += [int(i) for i in row if i >= 0.0]
            else:
                board.append([int(i) for i in row if i >= 0.0])

        return board

    @staticmethod
    def _num_players_from_state(state):
        return int(state[10])

    def _mcts(self, legal_actions, state):
        n = len(legal_actions)
        n_mc = self._compute_n_mc(n)
        outcomes = {action : [] for action in legal_actions}
        log_probs = {action : [] for action in legal_actions}

        for _ in range(n_mc):
            env = self._draw_env(legal_actions, state)
            action, log_prob, outcome = self._play_out(env, outcomes)
            outcomes[action].append(outcome)
            log_probs[action].append(log_prob)

        action, info = self._choose_action_from_outcomes(outcomes, log_probs)
        return action, info

    def _compute_n_mc(self, n_actions):
        return min(self.mc_max, self.mc_per_card * math.factorial(n_actions))

    def _draw_env(self, legal_actions, state):
        env = SechsNimmtEnv(self.num_players, self.num_rows, self.num_cards, self.threshold, self.include_summaries, verbose=False)
        board = self._board_from_state(state, flatten=False)
        hands = self._deal_hands(legal_actions)
        env.reset_to(board, hands)

        return env

    def _deal_hands(self, legal_actions):
        n = len(legal_actions)
        hands = [legal_actions.copy()]

        cards = self.available_cards.copy()
        np.random.shuffle(cards)
        cards = list(cards)
        for _ in range(self.num_players - 1):
            hands.append(sorted(cards[:n]))
            del cards[:n]

        return hands

    def _play_out(self, env, outcomes):
        states, all_legal_actions = env._create_states()
        states = self._tensorize(states)
        done = False
        outcome = 0.
        initial_action = None
        initial_log_prob = None

        while not done:
            actions, agent_infos = [], []
            for i, (state, legal_actions) in enumerate(zip(states, all_legal_actions)):
                action, log_prob = self._choose_action_mc(legal_actions, state, outcomes, first_move=(initial_action is None), opponent=(i > 0))
                actions.append(int(action))

                if initial_action is None:
                    initial_action = action
                    initial_log_prob = log_prob

            (next_states, next_all_legal_actions), next_rewards, done, _ = env.step(actions)
            next_states = self._tensorize(next_states)

            outcome += next_rewards[0]
            states = next_states
            all_legal_actions = next_all_legal_actions

        return initial_action, initial_log_prob, outcome

    def _choose_action_from_outcomes(self, outcomes, log_probs):
        best_action = list(outcomes.keys())[0]
        best_mean = - float("inf")

        for action, outcome in outcomes.items():
            if np.mean(outcome) > best_mean:
                best_action = action
                best_mean = np.mean(outcome)

        info = {"log_prob": log_probs[best_action][0]}

        logger.debug("AlphaAlmostZero thoughts:")
        for action, outcome in outcomes.items():
            chosen = 'x' if action == best_action else ' '
            logger.debug(f"  {chosen} {action + 1:>3d}: p = {np.exp(log_probs[action][0].detach().numpy()):.2f}, n = {len(outcome):>3d}, E[r] = {np.mean(outcome):>5.1f}")

        return best_action, info

    def _choose_action_mc(self, legal_actions, state, outcomes, first_move=True, opponent=False):
        raise NotImplementedError

    def _tensorize(self, inputs):
        return [torch.tensor(input).to(self.device, self.dtype) for input in inputs]


class MCSAgent(BaseMCAgent):
    """ Monte-Carlo search based on entirely random moves by all players """

    def learn(self, state, reward, action, done, next_state, next_reward, episode_end, num_episode, legal_actions, *args, **kwargs):
        pass

    def _choose_action_mc(self, legal_actions, state, outcomes, first_move=True, opponent=False):
        return np.random.choice(np.array(legal_actions, dtype=np.int), size=1)[0], torch.tensor(0.).to(self.device, self.dtype)


class PolicyMCSAgent(BaseMCAgent):
    """ Monte-Carlo search based on a learnable policy, similar to AlphaZero (but without a critic and without UCB) """

    def __init__(
        self,
        hidden_sizes=(100, 100,),
        activation=nn.ReLU(),
        r_factor=0.1,
        **kwargs
    ):
        super().__init__(**kwargs)

        self.r_factor = r_factor

        self.preprocessor = SechsNimmtStateNormalization(action=True)
        self.actor = MultiHeadedMLP(self.state_length + 1, hidden_sizes=hidden_sizes, head_sizes=(1,), activation=activation, head_activations=(None,))
        self.softmax = nn.Softmax(dim=0)

    def _choose_action_mc(self, legal_actions, state, outcomes, first_move=True, opponent=False):
        probs = self._compute_policy(legal_actions, state)

        cat = Categorical(probs)
        action_id = cat.sample()
        log_prob = cat.log_prob(action_id)
        action = legal_actions[action_id]

        return int(action), log_prob

    def _compute_policy(self, legal_actions, state):
        batch_states = []
        for action in legal_actions:
            action_ = torch.tensor([action]).to(self.device, self.dtype)
            batch_states.append(torch.cat((action_, state), dim=0).unsqueeze(0))
        batch_states = torch.cat(batch_states, dim=0)
        batch_states = self.preprocessor(batch_states)
        (probs,) = self.actor(batch_states)
        probs = self.softmax(probs).flatten()
        return probs

    def learn(self, state, reward, action, done, next_state, next_reward, episode_end, num_episode, legal_actions, *args, **kwargs):
        # Memorize step
        self.history.store(log_prob=kwargs["log_prob"], reward=reward * self.r_factor)

        # No further steps after each step
        if not episode_end or not self.training:
            return 0.

        # Gradient updates
        loss = self._train()

        # Reset memory for next episode
        self.history.clear()

        return loss

    def _train(self):
        # Roll out last episode
        rollout = self.history.rollout()
        log_probs = torch.stack(rollout["log_prob"], dim=0)

        # Compute loss
        loss = -torch.sum(log_probs)  # train policy to get closer to (deterministic) MCS choice

        # Gradient update
        self._gradient_step(loss)

        return loss.item()

    def _gradient_step(self, loss):
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()


class PUCTAgent(PolicyMCSAgent):
    def __init__(
        self,
        c_puct = 2.,
        temperature=None,
        **kwargs
    ):
        super().__init__(**kwargs)

        self.c_puct = c_puct
        self.temperature = temperature

    def _choose_action_mc(self, legal_actions, state, outcomes, first_move=True, opponent=False):
        # Opponents and subsequent moves are just drawn from policy, since we are almost certain to never have encountered that situation anyway
        if (not first_move) or opponent:
            return super()._choose_action_mc(legal_actions, state, outcomes, first_move, opponent)

        # For first move, use PUCT
        probs = self._compute_policy(legal_actions, state)
        pucts = self._compute_pucts(legal_actions, outcomes, probs)

        # Pick highest
        best_puct = - float("inf")
        choice = 0
        for i, puct in enumerate(pucts):
            if puct > best_puct:
                best_puct = puct
                choice = i

        return int(legal_actions[choice]), torch.log(probs[choice])

    def _compute_pucts(self, legal_actions, outcomes, probs):
        n = np.array([len(outcomes[action]) for action in legal_actions])
        n_total = sum(n)
        max_return, min_return, mean_return = self._normalize_q(outcomes)
        q = np.array([mean_return if not outcomes[action] else np.mean(outcomes[action]) for action in legal_actions])
        q = np.clip((q - min_return) / (max_return - min_return), 0., 1.)
        pucts = q + self.c_puct * probs.detach().numpy() * (n_total + 1.e-9) ** 0.5 / (1. + n)
        return pucts

    def _normalize_q(self, outcomes):
        all_outcomes = []
        for outcome in outcomes.values():
            all_outcomes += outcome

        if len(all_outcomes) < 10:
            return 0., -10., -5.  # TODO: don't hardcode this

        min_ = np.min(all_outcomes)
        max_ = np.max(all_outcomes)
        mean_ = np.median(all_outcomes)
        return max_, min_, mean_


    def _choose_action_from_outcomes(self, outcomes, log_probs):
        if self.temperature is None or self.temperature <= 1.e-12:
            return super()._choose_action_from_outcomes(outcomes, log_probs)

        # TODO: sampling from visited moves with probability ~ n(a)^(1/temperature)
        raise NotImplementedError
