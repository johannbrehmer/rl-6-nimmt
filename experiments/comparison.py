#! /usr/bin/env python

""" Top-level script for running experiments (if notebooks are not enough?) """

import sys
from matplotlib import pyplot as plt
import numpy as np
from multiprocessing import Pool
from configargparse import Namespace

sys.path.append("../")
from experiments.experiment import run_experiment

# Setup
env = "CartPole-v1"
algorithms = ["reinforce", "ac-mc", "acer", "dqn_prb", "ddqn_prb", "duelling_ddqn_prb"]
labels = ["REINFORCE", "MC actor-critic", "ACER", "DQN", "DDQN", "D3QN"]
runs = 5
parallel_jobs = 3
episodes = 1000


def make_config(algorithm, run):
    return Namespace(
        experiment=f"comparison_{env}_{algorithm}_{run}",
        algorithm=algorithm,
        env=env,
        episodes=episodes,
        steps=1000,
        dir="../",
        debug=False,
        silent=True,
        novideo=True,
        lr=1.0e-3,
        gamma=0.99,
        config=None,
        retrain=False,
    )


def make_agent_conf():
    return Namespace(history_length=200000, hidden_sizes=[100],)


jobs = len(algorithms) * runs


def worker(i):
    assert 0 <= i < jobs
    algorithm = algorithms[i % len(algorithms)]
    run = i // len(algorithms)

    print(f"Starting {algorithm} run {run}")
    config = make_config(algorithm, run)
    agent_config = make_agent_conf()
    try:
        run_experiment(None, config, agent_config)
    except Exception as e:
        print(f"{algorithm} run {run} errored:\n{e}")
        return np.nan * np.ones(episodes)

    try:
        returns = np.load(config.rewardprogressfile)
    except Exception as e:
        print(f"{algorithm} run {run} errored when loading rewards:\n{e}")
        return np.nan * np.ones(episodes)

    print(f"Done with {algorithm} run {run}")
    return returns


def running_mean(x, n):
    cumsum = np.cumsum(np.insert(x, 0, 0))
    return (cumsum[n:] - cumsum[:-n]) / float(n)


def plot_rewards(rewards, smoothen=50, cummax=False):
    rewards = np.array(rewards).reshape(runs, len(algorithms), -1)  # (algos, runs, time)
    if cummax:
        rewards = np.maximum.accumulate(rewards, axis=-1)
    reward_means = np.mean(rewards, axis=0)
    reward_stds = np.std(rewards, axis=0) / runs ** 0.5

    plt.figure(figsize=(6, 6))
    for i, (label, means, stds) in enumerate(zip(labels, reward_means, reward_stds)):
        ts = np.arange(smoothen / 2, episodes + 1 - smoothen / 2)
        means = running_mean(means, smoothen)
        stds = running_mean(stds, smoothen)
        plt.fill_between(ts, means - stds, means + stds, color=f"C{i}", alpha=0.15)
        plt.plot(ts, means, lw=1.5, c=f"C{i}", label=label)

    plt.legend(loc="lower right")
    plt.xlabel("Episode")
    plt.ylabel("Maximal reward" if cummax else "Reward")
    plt.xlim(0.0, None)
    plt.tight_layout()
    plt.savefig(f"comparison_{env}_max.pdf" if cummax else f"comparison_{env}.pdf")


if __name__ == "__main__":
    # TODO: use logging here (but still suppress the logging output from all other submodules)
    print("Hi!")
    pool = Pool(processes=parallel_jobs)
    rewards = pool.map(worker, range(jobs))
    print("Done with the training")

    print("Plotting...")
    plot_rewards(rewards)
    plot_rewards(rewards, cummax=True)

    print("That's it, have a nice day!")
