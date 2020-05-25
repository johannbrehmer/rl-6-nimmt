import torch
import numpy as np
import logging
from .env import SechsNimmtEnv

logger = logging.getLogger(__name__)


class GameSession:
    def __init__(self, *agents, device=torch.device("cpu"), dtype=torch.float):
        """ Initializes a game session, which consists of an arbitrary number of games between the given agents """

        self.device = device
        self.dtype = dtype
        self.agents = [agent.to(self.device, self.dtype) for agent in agents]
        self.num_agents = len(agents)
        self.env = SechsNimmtEnv(self.num_agents)
        self.results = []  # List of total scores (negative Hornochsen) for each game
        self.game = 0

        self._set_env_player_names()

    def play_game(self, render=False):
        """ Play one game, i.e. until one player hits 66 Hornochsen or whatever it is """

        states, all_legal_actions = self.env.reset()
        states = self._tensorize(states)
        done = False
        rewards = np.zeros(self.num_agents, dtype=np.int)
        scores = np.zeros(self.num_agents, dtype=np.int)

        if render:
            self.env.render()

        while not done:
            # Agent turns
            actions, agent_infos = [], []
            for agent, state, legal_actions in zip(self.agents, states, all_legal_actions):
                action, agent_info = agent(state, legal_actions=legal_actions)
                actions.append(int(action))
                agent_infos.append(agent_info)
            # TODO: gently enforce legality of actions by giving a negative reward and asking again

            # Environment steps
            (next_states, next_all_legal_actions), next_rewards, done, info = self.env.step(actions)
            next_states = self._tensorize(next_states)

            if render:
                self.env.render()

            # Learning
            for agent, action, state, next_state, reward, next_reward, agent_info, legal_actions, next_legal_actions, in zip(
                self.agents, actions, states, next_states, rewards, next_rewards, agent_infos, all_legal_actions, next_all_legal_actions
            ):
                agent.learn(
                    state=state,
                    legal_actions=legal_actions.copy(),
                    reward=reward,
                    action=action,
                    done=done,
                    next_state=next_state,
                    next_legal_actions=next_legal_actions.copy(),
                    next_reward=next_reward,
                    num_episode=self.game,
                    episode_end=done,
                    **agent_info,
                )

            scores += np.array(next_rewards)
            states = next_states
            all_legal_actions = next_all_legal_actions
            rewards = next_rewards

        self.results.append(scores)
        self.game += 1

    def _tensorize(self, inputs):
        return [torch.tensor(input).to(self.device, self.dtype) for input in inputs]

    def _set_env_player_names(self):
        names = []
        for agent in self.agents:
            try:
                names.append(agent.__name__)
            except:
                names.append(type(agent).__name__)
        self.env._player_names = names
