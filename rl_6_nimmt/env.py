import numpy as np
from gym import Env
from gym.spaces import Discrete, Box
import logging

logger = logging.getLogger(__name__)


class InvalidMoveException(Exception):
    pass


class SechsNimmtEnv(Env):
    """ OpenAI gym environment for the card game 6 Nimmt! """

    def __init__(self, num_players, num_rows=4, num_cards=104, threshold=6, include_summaries=True, player_names=None, verbose=True):
        super().__init__()

        assert num_players > 0
        assert num_rows > 0
        assert num_cards >= 10 * num_players + num_rows

        self._num_players = num_players
        self._num_rows = num_rows
        self._num_cards = num_cards
        self._threshold = threshold
        self._include_summaries = include_summaries
        self._player_names = player_names

        self._board = [[] for _ in range(self._num_rows)]
        self._hands = [[] for _ in range(self._num_players)]
        self._scores = np.zeros(self._num_players, dtype=np.int)

        self.action_space = Discrete(self._num_cards)
        self.reward_range = (-float("inf"), 0)
        self.metadata = {"render.modes": ["human"]}
        state_shape = (10 + 1 + int(self._include_summaries) * 3 * self._num_rows + self._num_rows * self._threshold,)
        self.observation_space = Box(low=-1.0, high=2.0, shape=state_shape, dtype=np.float)
        self.spec = None

        self.verbose = verbose

    def reset(self):
        """ Resets the state of the environment and returns an initial observation. """

        self._deal()
        self._scores = np.zeros(self._num_players, dtype=np.int)

        states = self._create_states()

        return states

    def reset_to(self, board, hands):
        """ Initializes a game for given board and hands """

        self._board = board
        self._hands = hands
        self._scores = np.zeros(self._num_players, dtype=np.int)

        states = self._create_states()

        return states

    def step(self, action):
        """ Environment step. action is actually a list of actions (one for each player). """

        assert len(action) == self._num_players
        for player, card in enumerate(action):
            self._check_move(player, card)

        rewards = self._play_cards(action)

        states = self._create_states()
        info = dict()
        done = self._is_done()

        return states, rewards, done, info

    def render(self, mode="human"):
        """ Report game progress somehow """

        logger.info("-" * 120)
        logger.info("Board:")
        for row, cards in enumerate(self._board):
            logger.info(f"  " + " ".join([self._format_card(card) for card in cards]) + "   _ " * (self._threshold - len(cards) - 1) + "   * ")
        logger.info("Players:")
        for player, (score, hand) in enumerate(zip(self._scores, self._hands)):
            self._player_name(player)
            logger.info(
                f"  {self._player_name(player)}: {score:>3d} Hornochsen, "
                + ("no cards " if len(hand) == 0 else "cards " + " ".join([self._format_card(card) for card in hand]))
            )
        if self._is_done():
            winning_player = np.argmin(self._scores)
            losing_player = np.argmax(self._scores)
            logger.info(f"The game is over! {self._player_name(winning_player)} wins, {self._player_name(losing_player)} loses. Congratulations!")
        logger.info("-" * 120)

    def _deal(self):
        """ Deals random cards to all players and initiates the game board """

        if self.verbose: logger.debug("Dealing cards")
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

        rewards = np.zeros(self._num_players, dtype=np.int)
        actions = [(card, player) for player, card in enumerate(cards)]
        actions = sorted(actions, key=lambda x: x[0])

        for card, player in actions:
            if self.verbose: logger.debug(f"{self._player_name(player)} plays card {card + 1}")
            row, replaced = self._find_row(card)
            self._board[row].append(card)
            self._hands[player].remove(card)

            if replaced or len(self._board[row]) >= self._threshold:
                rewards += self._score_row(row, player)

        return rewards

    def _find_row(self, card):
        """ Find which row a card has to go in """
        thresholds = [(row, cards[-1]) for row, cards in enumerate(self._board)]
        thresholds = sorted(thresholds, key=lambda x: -x[1])  # Sort by card threshold

        if card < thresholds[-1][1]:
            row = self._pick_row_to_replace()
            if self.verbose: logger.debug(f"  ...chooses to replace row {row + 1}")
            return row, True

        for row, threshold in thresholds:
            if card > threshold:
                return row, False

        raise ValueError(f"Cannot fit card {card} into thresholds {thresholds}")

    def _pick_row_to_replace(self):
        """ Picks which row should be replaces when a player undercuts the smalles open row """
        # TODO: In the long term this should be up to the agents.
        row_values = [self._row_value(cards, include_last=True) for cards in self._board]

        return np.argmin(row_values)

    def _score_row(self, row, player):
        """ Assigns points from a full row, and resets that row """
        cards = self._board[row]
        penalty = self._row_value(cards)
        if self.verbose: logger.debug(f"  ...and gains {penalty} Hornochsen")

        self._scores[player] += penalty
        rewards = np.zeros(self._num_players, dtype=np.int)
        rewards[player] -= penalty
        self._board[row] = [cards[-1]]

        return rewards

    def _create_states(self):
        """ Creates state tuple """

        game_state = self._create_game_state()
        player_states = []
        legal_actions = []

        for player in range(self._num_players):
            player_state, legal_action = self._create_agent_state(player)
            player_states.append(np.hstack((player_state, game_state)))
            legal_actions.append(legal_action)

        return player_states, legal_actions

    def _create_game_state(self):
        """ Builds game state """

        board_array = -np.ones((self._num_rows, self._threshold), dtype=np.int)
        for row, cards in enumerate(self._board):
            for i, card in enumerate(cards):
                board_array[row, i] = card

        if self._include_summaries:
            cards_per_row = np.array([len(cards) for cards in self._board], dtype=np.int)
            highest_per_row = np.array([cards[-1] for cards in self._board], dtype=np.int)
            score_per_row = np.array([self._row_value(cards, include_last=True) for cards in self._board], dtype=np.int)
            state = np.hstack(([self._num_players], cards_per_row, highest_per_row, score_per_row, board_array.flatten()))
        else:
            state = np.hstack(([self._num_players], board_array.flatten()))

        return state

    def _create_agent_state(self, player):
        """ Builds agent state for a given player """

        legal_actions = self._hands[player].copy()
        player_state = np.array(self._hands[player] + [-1 for _ in range(10 - len(legal_actions))], dtype=np.int)

        return player_state, legal_actions

    def _row_value(self, cards, include_last=False):
        """ Counts points (Hornochsen) in a row, excluding the last card """

        if include_last:
            return sum([self._card_value(c) for c in cards])
        elif len(cards) == 1:
            return 0
        else:
            return sum([self._card_value(c) for c in cards[:-1]])

    @staticmethod
    def _card_value(card):
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

    def _format_card(self, card):
        signs = {1: " ", 2: ".", 3: ":", 5: "+", 7: "#"}
        value = self._card_value(card)
        return f"{card + 1:>3d}{signs[value]}"

    def _is_done(self):
        """ Returns whether the game is over """

        return len(self._hands[0]) == 0

    def _player_name(self, player):
        if self._player_names is None:
            return f"Player {player + 1:d}"
        else:
            n = max([len(name) for name in self._player_names])
            return f"{self._player_names[player]:<{n}} (player {player + 1:d})"