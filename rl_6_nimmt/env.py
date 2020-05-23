import numpy as np
from gym import Env
from gym.spaces import Discrete, Tuple
import logging

logger = logging.getLogger(__name__)


class InvalidMoveException(Exception):
    pass


class SechsNimmtEnv(Env):
    action_space = Discrete(104)  # action = i means playing the card number i + 1
    observation_space = None

    # metadata = {'render.modes': []}
    reward_range = (-167, 0)
    # spec = None

    def __init__(self, num_players, num_rows=4, num_cards=104, threshold=6, include_summaries=True):
        super().__init__()

        assert num_players > 0
        assert num_rows > 0
        assert num_cards >= 10 * num_players + num_rows

        self._num_players = num_players
        self._num_rows = num_rows
        self._num_cards = num_cards
        self._threshold = threshold
        self._include_summaries = True

        self._board = [[] for _ in range(self._num_rows)]
        self._hands = [[] for _ in range(self._num_players)]
        self._scores = np.zeros(self._num_players, dtype=np.int)

    def reset(self):
        """ Resets the state of the environment and returns an initial observation. """
        self._deal()
        self._scores = np.zeros(self._num_players, dtype=np.int)
        states, done, info = self._create_states()
        return states

    def step(self, action):
        """ Environment step. action is actually a list of actions (one for each player). """
        assert len(action) == self._num_players
        for player, card in enumerate(action):
            self._check_move(player, card)
        rewards = self._play_cards(action)
        states, done, info = self._create_states()
        return states, rewards, done, info

    def render(self, mode="human"):
        """ Report game progress somehow """
        logger.info("")
        for player, (score, hand) in enumerate(zip(self._scores, self._hands)):
            logger.info(f"Player {player + 1:d}: {score:>3d} Hornochsen, cards " + " ".join([f"{card + 1:>3d}" for card in hand]))
        logger.info("Board:")
        for row, cards in enumerate(self._board):
            logger.info(f"  [{row + 1:d}]: " + " ".join([f"{card + 1:>3d}" for card in cards]))
        logger.info("")

    def _deal(self):
        """ Deals random cards to all players and initiates the game board """
        cards = np.arange(0, self._num_cards, 1, dtype=np.int)
        np.random.shuffle(cards)
        cards = list(cards)

        for player in range(self._num_players):
            self._hands[player] = sorted(cards[:10])  # pop() does not support multiple indices, does it?
            del cards[:10]

        for row in range(self._num_rows):
            self._board[row] = [cards.pop()]

    def _check_move(self, player, card):
        """ Check legality of a move and raise an exception otherwise"""
        if card not in self._hands[player]:
            raise InvalidMoveException(f"Player {player + 1} tried to play card {card + 1}, but their hand is {self._hands[player]}")

    def _play_cards(self, cards):
        """ Given one played card per player, play the cards, score points, and update the game """
        rewards = np.zeros(self._num_players)
        associated_players = {card: player for player, card in enumerate(cards)}
        for card in sorted(cards):
            row, replaced = self._find_row(card)
            self._board[row].append(card)
            self._hands[associated_players[card]].remove(card)

            if replaced or len(self._board[row]) >= self._threshold:
                rewards += self._score_row(row, associated_players[card])

        return rewards

    def _find_row(self, card):
        """ Find which row a card has to go in """
        thresholds = [(row, cards[-1]) for row, cards in enumerate(self._board)]
        thresholds = sorted(thresholds, key=lambda x: -x[1])  # Sort by card threshold

        if card < thresholds[-1][1]:
            return self._pick_row_to_replace(), True

        for row, threshold in thresholds:
            if card > threshold:
                return row, False

        raise ValueError(f"Cannot fit card {card} into thresholds {thresholds}")

    def _pick_row_to_replace(self):
        """ Picks which row should be replaces when a player undercuts the smalles open row """
        # TODO: In the long term this should be up to the agents.
        row_values = [self._row_value(cards) for cards in self._board]
        return np.argmin(row_values)

    def _score_row(self, row, player):
        """ Assigns points from a full row, and resets that row """
        cards = self._board[row]
        self._scores[player] += self._row_value(cards)
        self._board[row] = [cards[-1]]
        rewards = np.zeros(self._num_players)
        rewards[player] -= self._row_value(cards)
        return rewards

    def _create_states(self):
        done = len(self._hands[0]) == 0
        states = [self._create_game_state()]
        for player in range(self._num_players):
            states.append(self._create_agent_state(player))
        info = {}
        return states, done, info

    def _create_game_state(self):
        """ Builds game state """
        board_array = -np.ones((self._num_rows, self._threshold))
        for row, cards in enumerate(self._board):
            for i, card in enumerate(cards):
                board_array[row, i] = card

        if self._include_summaries:
            cards_per_row = np.array([len(cards) for cards in self._board])
            highest_per_row = np.array([cards[-1] for cards in self._board])
            score_per_row = np.array([self._row_value(cards, include_last=True) for cards in self._board])
            state = np.hstack((cards_per_row, highest_per_row, score_per_row, board_array.flatten()))
        else:
            state = np.flatten(board_array)

        return state

    def _create_agent_state(self, player):
        """ Builds agent state for a given player """
        return self._hands[player]

    def _row_value(self, cards, include_last=False):
        """ Counts points (Hornochsen) in a row, excluding the last card """
        if include_last:
            return sum([self._card_value(c) for c in cards])
        elif len(cards) == 1:
            return 0
        else:
            return sum([self._card_value(c) for c in cards[:-1]])

    def _card_value(self, card):
        """ Returns the points (Hornochsen) on a single card """
        assert 0 <= card < 104
        if card + 1 == 55:
            return 7
        elif (card + 1) % 11 == 0:  # 11, 22, ..., 99; but not 55
            return 5
        elif (card + 1) % 10 == 0:  # 10, 20, 30, ..., 100
            return 3
        elif (card + 1) % 10 == 5:  # 5, 15, ..., 95, but not 55
            return 2
        else:
            return 1
