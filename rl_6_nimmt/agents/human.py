from .base import Agent
import logging

logger = logging.getLogger(__name__)


class Human(Agent):
    """ Agent that makes totally random decisions """

    def __init__(self, name="Human", env=None, *args, **kwargs):
        super().__init__(env, *args, **kwargs)
        self.__name__ = name  # This is a terrible idea

    def forward(self, state, legal_actions, **kwargs):
        prompt = (
            f"It is your turn, {self.__name__}! You have the following cards: "
            + " ".join([f"{card + 1:>3d}" for card in legal_actions])
            + ". Choose one to play!"
        )

        action = -1
        while action not in legal_actions:
            action = input(prompt)
            try:
                action = int(action) - 1
            except:
                logger.error("Input in wrong format, please try again.")
            prompt = "You don't have that card. Please pick one of your cards: " + " ".join([f"{card + 1:>3d}" for card in legal_actions])

        return action, {}

    def learn(self, *args, **kwargs):
        return 0.0
