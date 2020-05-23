#! /usr/bin/env python

import gym
import logging
import sys
from line_profiler import LineProfiler

sys.path.append("../")
from reinforcing_fun.agents import ReinforceAgent
from reinforcing_fun.training import ReinforceTrainer

logging.basicConfig(format="%(asctime)-5.5s %(name)-20.20s %(levelname)-7.7s %(message)s", datefmt="%H:%M", level=logging.INFO)

env = gym.make("Acrobot-v1")
agent = ReinforceAgent(env, gamma, hidden_sizes=(64, 64), state_length=len(env.reset()))
trainer = ReinforceTrainer(env, agent)

lp = LineProfiler()
lp.add_function(trainer._finish_episode)
lp.add_function(trainer._episode)
lp_wrapper = lp(trainer.train)

rewards, lenghts, losses = lp_wrapper(num_episodes=100)
lp.print_stats()
