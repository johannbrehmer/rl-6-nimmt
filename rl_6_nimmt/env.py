import numpy as np
from gym import Env
from gym.spaces import Discrete, Tuple


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

        self._num_players = num_players
        self._num_rows = num_rows
        self._num_cards = num_cards
        self._threshold = threshold
        self._include_summaries = True

        self._board = [[] for _ in range(self._num_rows)]
        self._hands = [[] for _ in range(self._num_players)]
        self._scores = np.zeros(self._num_players)

    def reset(self):
        """Resets the state of the environment and returns an initial observation.
        Returns:
            observation (object): the initial observation.
        """

        self._deal()
        self._scores = np.zeros(self._num_players)

    def step(self, action):
        """Run one timestep of the environment's dynamics. When end of
        episode is reached, you are responsible for calling `reset()`
        to reset this environment's state.
        Accepts an action and returns a tuple (observation, reward, done, info).
        Args:
            action (object): an action provided by the agent
        Returns:
            observation (object): agent's observation of the current environment
            reward (float) : amount of reward returned after previous action
            done (bool): whether the episode has ended, in which case further step() calls will return undefined results
            info (dict): contains auxiliary diagnostic information (helpful for debugging, and sometimes learning)
        """
        raise NotImplementedError

    def render(self, mode='human'):
        """Renders the environment.
        The set of supported modes varies per environment. (And some
        environments do not support rendering at all.) By convention,
        if mode is:
        - human: render to the current display or terminal and
          return nothing. Usually for human consumption.
        - rgb_array: Return an numpy.ndarray with shape (x, y, 3),
          representing RGB values for an x-by-y pixel image, suitable
          for turning into a video.
        - ansi: Return a string (str) or StringIO.StringIO containing a
          terminal-style text representation. The text can include newlines
          and ANSI escape sequences (e.g. for colors).
        Note:
            Make sure that your class's metadata 'render.modes' key includes
              the list of supported modes. It's recommended to call super()
              in implementations to use the functionality of this method.
        Args:
            mode (str): the mode to render with
        Example:
        class MyEnv(Env):
            metadata = {'render.modes': ['human', 'rgb_array']}
            def render(self, mode='human'):
                if mode == 'rgb_array':
                    return np.array(...) # return RGB frame suitable for video
                elif mode == 'human':
                    ... # pop up a window and render
                else:
                    super(MyEnv, self).render(mode=mode) # just raise an exception
        """
        raise NotImplementedError

    def close(self):
        """Override close in your subclass to perform any necessary cleanup.
        Environments will automatically close() themselves when
        garbage collected or when the program exits.
        """
        pass

    def _deal(self):
        """ Deals random cards to all players and initiates the game board """
        cards = np.arange(0, self._num_cards, 1, dtype=np.int)
        np.random.shuffle(cards)
        cards = list(cards)

        for player in range(self._num_players):
            self._hands[player] = cards[:10]  # pop() does not support multiple indices, does it?
            del cards[:10]

        for row in self._num_rows:
            self._board[row] = [cards.pop()]

    def _check_move(self, player, card):
        if card not in self._hands[player]:
            raise InvalidMoveException(f"Player {player + 1} tried to play card {card + 1}, but their hand is {self.hands[player]}")

    def _play_cards(self, cards):
        """ Given one played card per player, play the cards, score points, and update the game """
        rewards = np.zeros(self._num_players)
        associated_players = {card: player for player, card in enumerate(cards)}
        for card in sorted(cards):
            row = self._find_row(card)
            self._board[row].append(card)

            if len(self._board[row]) >= self._threshold:
                rewards += self._score_row(row, associated_players[card])

        return rewards

    def _find_row(self, card):
        """ Find which row a card has to go in """
        thresholds = [(row, cards[-1]) for row, cards in enumerate(self._board)]
        thresholds = sorted(thresholds, key=lambda x : x[1])  # Sort by card threshold

        if card < thresholds[0][1]:
            return self._pick_row_to_replace()
        for row, threshold in thresholds:
            if card > threshold:
                return row

        raise ValueError(f"Cannot fit card {card} into thresholds {thresholds}")

    def _pick_row_to_replace(self):
        """ Picks which row should be replaces when a player undercuts the smalles open row """
        # TODO: In the long term this should be up to the agents.
        row_values = [self._row_values(row) for row in self._board]
        return np.argmin(row_values)

    def _score_row(self, row, player):
        """ Assigns points from a full row, and resets that row """
        cards = self._board[row]
        self._scores[player] += self._row_value(cards)
        self._board[row] = [cards[-1]]
        rewards = np.zeros(self._num_players)
        rewards[player] -= self._row_value(cards)
        return rewards

    def _create_game_state(self):
        """ Builds game state """
        raise NotImplementedError

    def _create_agent_state(self, player):
        """ Builds agent state for a given player """
        raise NotImplementedError

    def _row_value(self, cards):
        """ Counts points (Hornochsen) in a row, excluding the last card """
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
