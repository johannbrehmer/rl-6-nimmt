import logging
import sys

sys.path.append("../")

from rl_6_nimmt import Tournament, GameSession
from rl_6_nimmt.agents import DrunkHamster, BatchedACERAgent

logging.basicConfig(format="%(message)s",level=logging.DEBUG)
for name in logging.root.manager.loggerDict:
    if not "rl_6_nimmt" in name:
        logging.getLogger(name).setLevel(logging.WARNING)

acer = BatchedACERAgent(warmup=0)
acer.train()
hamster = DrunkHamster()

session = GameSession(acer, hamster)

session.play_game(render=True)
