from ..env import SechsNimmtEnv
from .base import Agent
import math
import numpy as np


class MCTSAgent(Agent):
    def __init__(
        self,
        handsize=10,
        num_rows=4,
        num_cards=104,
        threshold=6,
        mc_per_card=10,
        mc_max=10000,
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
            action = legal_actions[0]
        else:
            action = self._mcts(legal_actions, state)

        return action, {}

    def learn(self, state, reward, action, done, next_state, next_reward, episode_end, num_episode, legal_actions, *args, **kwargs):
        pass

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

        for _ in range(n_mc):
            env = self._draw_env(legal_actions, state)
            action, outcome = self._play_out(env)
            outcomes[action].append(outcome)

        action = self._choose_action_from_outcomes(outcomes)
        return action

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

    def _play_out(self, env):
        states, all_legal_actions = env._create_states()
        done = False
        outcome = 0.
        first_player_action = None

        while not done:
            actions, agent_infos = [], []
            for i, (state, legal_actions) in enumerate(zip(states, all_legal_actions)):
                action = self._choose_action_mc(legal_actions, state, opponent=(i > 0))
                actions.append(int(action))

            if first_player_action is None:
                first_player_action = actions[0]

            (next_states, next_all_legal_actions), next_rewards, done, _ = env.step(actions)

            outcome += next_rewards[0]
            states = next_states
            all_legal_actions = next_all_legal_actions

        return first_player_action, outcome

    def _choose_action_from_outcomes(self, outcomes):
        best_action = list(outcomes.keys())[0]
        best_mean = - float("inf")

        for action, outcome in outcomes.items():
            if np.mean(outcome) > best_mean:
                best_action = action
                best_mean = np.mean(outcome)

        return best_action

    def _choose_action_mc(self, legal_actions, state, opponent=False):
        raise NotImplementedError


class RandomMCTSAgent(MCTSAgent):
    def _choose_action_mc(self, legal_actions, state, opponent=False):
        return np.random.choice(np.array(legal_actions, dtype=np.int), size=1)[0]
