import gym
import logging
from reinforcing_fun.agents import DQNVanilla
from reinforcing_fun.train import Trainer
import torch
from torch.utils.tensorboard import SummaryWriter
import math

import numpy as np
from reinforcing_fun.utils.various import timeit

logging.basicConfig(format="%(asctime)-5.5s %(name)-20.20s %(levelname)-7.7s %(message)s", datefmt="%H:%M", level=logging.INFO)

env = gym.make("CartPole-v1")
# agent = StateValueActorCriticAgent(env, 4)
# trainer = TD0ActorCriticTrainer(env, agent)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


hp = {
    "retrain_interval": 4,  # steps between local qnet and target network
    "retrain_batch_size": 64,
    "replay_buffer_size": int(2e5),
    "tau": 1e-2,  # for soft update of target parameters, 1 would be a hard target update
    "gamma": 0.999,
    "lr": 1e-3,
    "max_episodes": 2000,
    "use_target_net": True,
}

comment_string = "DQN_target_net" if hp["use_target_net"] else "DQN_single"
summary_writer = SummaryWriter(comment=comment_string)


def eps_func_step(episode):
    rel_val = episode / hp["max_episodes"]
    min_val = 0.05
    max_val = 0.99
    rel_start = 0.0
    rel_end = 0.5
    if rel_val < rel_start:
        return max_val
    elif rel_end >= rel_val >= rel_start:
        m = (min_val - max_val) / (rel_end - rel_start)
        b = max_val - rel_start * m
        return rel_val * m + b
    else:
        return min_val


def eps_func_decay(episode):
    rel_val = episode
    decay_rate = 0.0025
    eps = 1 * math.exp(-rel_val * decay_rate)
    eps = max(eps, 0.05)
    return eps


hp["eps_func"] = eps_func_decay
# @timeit


def dqn_debug_callback_eps(trainer, **kwargs):
    agent_info = kwargs["agent_info"]
    if trainer.episode % 10 == 0:
        trainer.summary_writer.add_scalar("debug/eps", agent_info["eps"], trainer.episode)


def dqn_debug_callback_weights(trainer, **kwargs):
    agent = trainer.agent
    if trainer.episode % 10 == 0:
        for x in agent.dqn_net_local.named_parameters():
            if "head_nets" in x[0]:
                trainer.summary_writer.add_histogram("networks/local_" + x[0], x[1], trainer.episode)
        for x in agent.dqn_net_target.named_parameters():
            if "head_nets" in x[0]:
                trainer.summary_writer.add_histogram("networks/target_" + x[0], x[1], trainer.episode)


agent = DQNVanilla(
    env=env,
    hidden_sizes=(64, 64),
    device=device,
    summary_writer=summary_writer,
    optim_kwargs={"lr": hp["lr"]},
    retrain_interval=hp["retrain_interval"],
    eps_func=hp["eps_func"],
    use_target_net=hp["use_target_net"],
    minibatch=hp["retrain_batch_size"],
    history_length=hp["replay_buffer_size"],
    tau=hp["tau"],
)

for key, val in hp.items():
    summary_writer.add_text(key, str(val))
    logging.info(f"{key}: {str(val)}")


trainer = Trainer(env, agent, device=device, dtype=torch.float, summary_writer=summary_writer)


logging.info("Starting training: Using following Hyperparameters:")
for key, val in hp.items():
    logging.info(f"{key}: {str(val)}")

rewards, lenghts, losses = trainer.train(
    num_episodes=hp["max_episodes"], gamma=hp["gamma"], device=device, dtype=torch.float, callbacks=[dqn_debug_callback_weights]
)

# metrics = {"hparam/max_reward": np.max(rewards), "hparam/mean_loss": np.mean(losses)}

summary_writer.close()
