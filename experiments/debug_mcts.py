import logging
import sys

sys.path.append("../")

from rl_6_nimmt import GameSession
from rl_6_nimmt.agents import DrunkHamster, RandomMCTSAgent

logging.basicConfig(format="%(message)s",level=logging.DEBUG)
for name in logging.root.manager.loggerDict:
    if not "rl_6_nimmt" in name:
        logging.getLogger(name).setLevel(logging.WARNING)

mcts = RandomMCTSAgent()
hamster = DrunkHamster()

session = GameSession(mcts, hamster)

session.play_game(render=True)
