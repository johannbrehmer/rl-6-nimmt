import logging
import torch
import gym
import sys

import logging

sys.path.append("../")
from reinforcing_fun.agents import AGENTS
from reinforcing_fun.train import Trainer
from line_profiler import LineProfiler


def set_up_torch(config):
    """ pyTorch preparations """

    # dtype and device
    config.dtype = torch.float
    if torch.cuda.is_available():
        logger.info("Running on GPU")
        config.device = torch.device("cuda")
        torch.set_default_tensor_type("torch.cuda.FloatTensor")
    else:
        logger.info("Running on CPU")
        config.device = torch.device("cpu")
        torch.set_default_tensor_type("torch.FloatTensor")


# Set up logging


class Config:
    pass


if __name__ == "__main__":
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    log_formatter = logging.Formatter("%(asctime)-5.5s %(levelname)-7.7s %(message)s", datefmt="%H:%M")

    std_handler = logging.StreamHandler(sys.stdout)
    std_handler.setFormatter(log_formatter)
    logger.addHandler(std_handler)

    logger.info("Hi!")
    logger.info("Starting line profiling %s with algorithm %s on environment")
    config = Config()
    config.device = "cpu"
    config.dtype = torch.float
    config.env = "CartPole-v1"
    config.lr = 1e-4
    summary_writer = None
    set_up_torch(config)

    env = gym.make(config.env)
    agent = AGENTS["duelling_ddqn_pbr"](
        env,
        optim_kwargs={"lr": config.lr},
        retrain_interval=4,
        minibatch=64,
        history_length=int(2 ** 16),
        tau=1e-2,
        device=config.device,
        dtype=config.dtype,
        summary_writer=None,
    )
    lp = LineProfiler()
    # lp.add_function(agent.learn)
    lp.add_function(agent._learn)
    lp.add_function(agent.history.sample)
    lp.add_function(agent.history.get_samples)
    # lp.add_function(agent.history.tree.update)

    # Training
    trainer = Trainer(env, agent, device=config.device, dtype=config.dtype, summary_writer=summary_writer)
    trainer.train = lp(trainer.train)
    training_progress = trainer.train(num_episodes=100)
    env.close()
    lp.print_stats()
