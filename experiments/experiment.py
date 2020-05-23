#! /usr/bin/env python

""" Top-level script for running experiments (if notebooks are not enough?) """

import logging
import configargparse
import numpy as np
import torch
import gym
import sys
import os
from matplotlib import pyplot as plt
from torch.utils.tensorboard import SummaryWriter
import datetime

sys.path.append("../")
from reinforcing_fun.agents import AGENTS, DDQN_METHODS, POLICY_METHODS, NSTEP_METHODS, NOISY_METHODS
from reinforcing_fun.train import Trainer


def parse_args():
    """ Parses command line arguments for the training """
    # TODO place default arguments for algorithms all in agents and pass a kwargs dict?

    p_top_level = configargparse.ArgumentParser()
    p_agent = configargparse.ArgumentParser()

    p_top_level.add_argument("experiment", type=str, help="Experiment name")
    p_top_level.add_argument("algorithm", type=str, choices=AGENTS.keys(), help="Algorithm")
    p_top_level.add_argument("env", type=str, help="Environment")
    p_top_level.add_argument("--retrain", action="store_true", help="Overwrite existing model and train again")
    p_top_level.add_argument("--novideo", action="store_true", help="Skip generating a video")
    p_top_level.add_argument("--config", is_config_file=True, type=str, help="Config file")
    p_top_level.add_argument("--dir", type=str, default="../", help="Base directory of repo")
    p_top_level.add_argument("--debug", action="store_true", help="Activate debug mode")
    p_top_level.add_argument("--silent", action="store_true", help="Do not print logging output to stdout")
    p_top_level.add_argument("--episodes", type=int, default=2000, help="Number of episodes for training")
    p_top_level.add_argument("--steps", type=int, default=1000, help="Maximum number of steps per episode")
    p_top_level.add_argument("--lr", type=float, default=1.0e-2, help="Learning rate")
    p_top_level.add_argument("--gamma", type=float, default=0.99, help="Discount factor for Bellmann-equation")
    top_level_conf, unknown_args = p_top_level.parse_known_args()

    p_agent.add_argument("--history-length", type=int, default=200000, help="Replay buffer length for DQN")
    p_agent.add_argument("--hidden-sizes", type=int, nargs="+", default=[64,], help="size and number of hidden layers in MLP")
    if top_level_conf.algorithm in POLICY_METHODS:
        p_agent.add_argument("--actorweight", type=float, default=1.0, help="Multiplies the actor loss in actor-critic models")
        p_agent.add_argument("--criticweight", type=float, default=1.0, help="Multiplies the critic loss in actor-critic models")
        p_agent.add_argument("--entropyweight", type=float, default=0.0, help="Weights an entropy regularization term")
    if top_level_conf.algorithm in NOISY_METHODS:
        p_agent.add_argument("--noisy-init-sigma", type=float, default=1, help="NoisyDQN inital sigma noisy for noisy layers")
    if top_level_conf.algorithm in POLICY_METHODS:
        p_agent.add_argument("--actorweight", type=float, default=1.0, help="Multiplies the actor loss in actor-critic models")
        p_agent.add_argument("--criticweight", type=float, default=1.0, help="Multiplies the critic loss in actor-critic models")
        p_agent.add_argument("--entropyweight", type=float, default=0.0, help="Weights an entropy regularization term")
        p_agent.add_argument("-n", type=int, default=3, help="Number of steps in n-step returns")
    if top_level_conf.algorithm in DDQN_METHODS:
        p_agent.add_argument("--retrain-interval", type=int, default=4, help="DQN retrain interval")
        p_agent.add_argument("--tau", type=float, default=1.0e-2, help="DQN tau parameter")
        p_agent.add_argument("--minibatch", type=int, default=64, help="DQN minibatch size")
    if top_level_conf.algorithm in NSTEP_METHODS:
        p_agent.add_argument("--n-steps", type=int, default=5, help="Steps to evaluate (makes only sense for n-step types")

    agent_conf = p_agent.parse_args(unknown_args)
    return p_top_level, top_level_conf, agent_conf


def set_up_experiment(parser, config):
    """ Sets up filenames and logger, and checks if a model is already present """

    # Create filenames
    config.expdir = os.path.join(config.dir, "experiments", "data", config.experiment)
    config.modelfile = os.path.join(config.expdir, "{}.pt".format(config.experiment))
    config.logfile = os.path.join(config.expdir, "{}.log".format(config.experiment))
    config.configfile = os.path.join(config.expdir, "{}.config".format(config.experiment))
    config.trainingfigfile = os.path.join(config.expdir, "{}_training.pdf".format(config.experiment))
    config.videodir = os.path.join(config.expdir, "videos")
    config.tensorboarddir = os.path.join(config.dir, "experiments", "data", "tensorboard")  # One common directory for all tensorboard logs makes life easier
    config.rewardprogressfile = os.path.join(config.expdir, "{}_rewards.npy".format(config.experiment))

    # Create experiment folder and subfolders
    os.makedirs(config.expdir, exist_ok=True)
    os.makedirs(config.videodir, exist_ok=True)

    # Set up logging
    logger = logging.getLogger()
    logger.setLevel(logging.DEBUG if config.debug else logging.INFO)
    log_formatter = logging.Formatter("%(asctime)-5.5s %(levelname)-7.7s %(message)s", datefmt="%H:%M")

    file_handler = logging.FileHandler(config.logfile)
    file_handler.setFormatter(log_formatter)
    logger.addHandler(file_handler)
    if not config.silent:
        std_handler = logging.StreamHandler(sys.stdout)
        std_handler.setFormatter(log_formatter)
        logger.addHandler(std_handler)

    # Set up summary writer
    ts = datetime.datetime.now().strftime("%b%d__%H-%M-%S_")

    tb_log_dir = os.path.join(config.tensorboarddir, ts + config.experiment)
    summary_writer = SummaryWriter(log_dir=tb_log_dir)

    # Check if a model already exists in this folder
    model_exists = os.path.exists(config.modelfile)

    # Save setup to config file
    if parser is not None:
        parser.write_config_file(config, [config.configfile])

    return model_exists, logger, summary_writer


def set_up_torch(config, logger):
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

    # Bug fix related to some num_workers > 1 and CUDA. Bad things happen otherwise! Not needed as long as we don't have workers.
    # torch.multiprocessing.set_start_method("spawn", force=True)


def train_model(agent, env, config, logger, summary_writer, agent_config):
    """ Trains model """
    logger.info("Training model for %s episodes", config.episodes)

    # Training
    trainer = Trainer(env, agent, device=config.device, dtype=config.dtype, summary_writer=summary_writer)

    training_progress = trainer.train(
        num_episodes=config.episodes, silent=config.silent, **vars(agent_config)
    )  # TODO do we need here also critique factor and all these rather agent specific kwargs?
    env.close()

    return agent, training_progress


def create_agent_and_env(config, logger, summary_writer, agent_config):
    logger.info(f"Setting up environment {config.env}")
    env = gym.make(config.env)

    logger.info(f"Setting up agent {config.algorithm}")
    agent = AGENTS[config.algorithm](
        env, config.gamma, optim_kwargs={"lr": config.lr}, device=config.device, dtype=config.dtype, summary_writer=summary_writer, **vars(agent_config)
    )
    return agent, env


def load_model(agent, env, config, logger):
    """ Loads model state dict from file """

    logger.info("Loading model from %s", config.modelfile)
    agent.load_state_dict(torch.load(config.modelfile, map_location=config.device))

    return agent


def save_model(model, config, logger):
    """ Saves model state dict to file """

    logger.info("Saving model to %s", config.modelfile)
    torch.save(model.state_dict(), config.modelfile)


def save_returns(training_progress, config, logger):
    """ Saves model state dict to file """

    logger.info("Saving returns to %s", config.rewardprogressfile)
    np.save(config.rewardprogressfile, training_progress[0])


def plot_training_progress(training_progress, config, logger):
    """ Plots training progress (rewards, losses, etc) """

    # TODO: do we want to keep this if we also have TensorBoard? It's a bit redundant, but I like that I get a pdf directly...

    if training_progress is None:
        return

    logger.info("Plotting training progress and saving at %s", config.trainingfigfile)

    if len(training_progress[2].shape) > 1:
        quantities = [training_progress[0], training_progress[1]] + list(training_progress[2].T)
        labels = ["Reward", "Episode length"] + [f"Loss {i}" for i in range(training_progress[2].shape[1])]
    else:
        quantities = [training_progress[0], training_progress[1], training_progress[2] / training_progress[1]]
        labels = ["Reward", "Episode length", "Loss"]

    num_panels = len(quantities)
    ticks = np.linspace(0.0, config.episodes, 6)

    def running_mean(x, n):
        cumsum = np.cumsum(np.insert(x, 0, 0))
        return (cumsum[n:] - cumsum[:-n]) / float(n)

    plt.figure(figsize=(6.0, 3.0 * num_panels))
    for panel, (quantity, label) in enumerate(zip(quantities, labels)):
        plt.subplot(num_panels, 1, panel + 1)
        plt.scatter(np.arange(1, config.episodes + 1), quantity, c="C3", s=4.0, alpha=0.25, rasterized=True)
        plt.plot(np.arange(25, config.episodes - 24), running_mean(quantity, 50), lw=1.5, c="C3", zorder=1)
        plt.ylabel(label)
        plt.xticks(ticks, labels=["" for _ in ticks] if panel < num_panels - 1 else None)
    plt.tight_layout()
    plt.savefig(config.trainingfigfile)


def make_video(model, config, logger):
    """ Makes a video of the model fighting with the environment """

    logger.info("Rendering video and saving to %s", config.videodir)

    # Set up environment
    env = gym.make(config.env)
    env = gym.wrappers.Monitor(env, config.videodir, video_callable=lambda episode_id: True, force=True)
    model.eval()

    for _ in range(10):
        state = env.reset()

        for _ in range(config.steps):
            env.render()
            state_ = torch.from_numpy(state).to(dtype=config.dtype, device=config.device)
            action = model(state_)[0]
            state, reward, done, info = env.step(action)

            if done:
                for _ in range(10):
                    env.render()
                break

    env.close()


def run_experiment(parser, config, agent_config):
    model_exists, logger, summary_writer = set_up_experiment(parser, config)

    logger.info("Hi!")
    logger.info("Starting experiment %s with algorithm %s on environment %s", config.experiment, config.algorithm, config.env)
    logger.info(" ")
    logger.info("Agent options:")
    for k, v in vars(agent_config).items():
        logger.info(f"{k}: {v}")
    logger.info(" ")
    logger.info("Top level options:")
    for k, v in vars(config).items():
        logger.info(f"  {k}: {v}")

    set_up_torch(config, logger)
    model, env = create_agent_and_env(config, logger, summary_writer, agent_config)

    if model_exists and not config.retrain:
        model = load_model(model, env, config, logger)
    else:
        model, training_progress = train_model(model, env, config, logger, summary_writer, agent_config)
        save_model(model, config, logger)
        save_returns(training_progress, config, logger)
        try:
            plot_training_progress(training_progress, config, logger)
        except ValueError as e:
            logger.exception(f"Could not create plots")

    if not config.novideo:
        make_video(model, config, logger)

    logger.info("All done! Have a nice day!")


if __name__ == "__main__":
    parser, config, agent_config = parse_args()
    run_experiment(parser, config, agent_config)
