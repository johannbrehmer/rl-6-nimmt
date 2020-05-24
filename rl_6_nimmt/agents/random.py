from .base import Agent
import numpy as np


class DrunkHamster(Agent):
    """ Agent that makes totally random decisions """

    def forward(self, state, legal_actions, **kwargs):
        action = np.random.choice(np.array(legal_actions, dtype=np.int), size=1)[0]
        return action, {}

    def learn(self, *args, **kwargs):
        return 0.0
