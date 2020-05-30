import logging
import sys

sys.path.append("../")

from rl_6_nimmt import Tournament, GameSession
from rl_6_nimmt.agents import DrunkHamster, BatchedACERAgent, DQNVanilla, Noisy_D3QN_PRB_NStep

logging.basicConfig(format="%(message)s",level=logging.DEBUG)
for name in logging.root.manager.loggerDict:
    if not "rl_6_nimmt" in name:
        logging.getLogger(name).setLevel(logging.WARNING)


dqn = Noisy_D3QN_PRB_NStep(history_length=int(1e5))
#dqn = DQNVanilla()
dqn.train()
hamster = DrunkHamster()

session = GameSession(dqn, hamster)

session.play_game(render=True)
