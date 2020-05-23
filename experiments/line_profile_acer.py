#! /usr/bin/env python

import sys
from configargparse import Namespace

sys.path.append("../")
from experiments.experiment import set_up_experiment, set_up_torch, create_agent_and_env
from reinforcing_fun.train import Trainer
from line_profiler import LineProfiler


if __name__ == "__main__":
    parser = None
    config = Namespace(
        experiment="profile_acer",
        algorithm="acer",
        env="CartPole-v1",
        episodes=50,
        steps=500,
        lr=1.0e-3,
        criticfactor=1.0,
        entropyfactor=0.0,
        history_length=600,
        retraininterval=4,
        tau=1.0e-2,
        gamma=0.99,
        minibatch=4,
        retrain=True,
        config=None,
        dir="../",
        debug=False,
        silent=False,
        novideo=True,
    )
    agent_config = Namespace()

    model_exists, logger, summary_writer = set_up_experiment(parser, config)

    logger.info("Hi! Setting up line profiling")
    set_up_torch(config, logger)
    model, env = create_agent_and_env(config, logger, summary_writer, agent_config)
    model.train()
    trainer = Trainer(env, model, device=config.device, dtype=config.dtype, summary_writer=summary_writer)

    lp = LineProfiler()
    lp.add_function(model.learn)
    lp.add_function(model._train)

    logger.info("Starting training")
    trainer.train = lp(trainer.train)
    trainer.train(num_episodes=config.episodes, critic_factor=config.criticfactor, entropy_factor=config.entropyfactor, silent=config.silent)
    env.close()

    logger.info("Line profiling results:")
    lp.print_stats()

    logger.info("All done! Have a nice day!")
