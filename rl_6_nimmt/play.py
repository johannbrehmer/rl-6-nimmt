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

        states = self.env.reset()
        game_state = states[0].copy()
        agent_states = [state.copy() for state in states[1:]]
        done = False
        rewards = np.zeros(self.num_agents, dtype=np.int)
        scores = np.zeros(self.num_agents, dtype=np.int)

        if render:
            self.env.render()

        while not done:
            # Agent turns
            state_tensor = torch.tensor(game_state).to(self.device, self.dtype)
            actions, agent_infos = [], []
            for agent, agent_state in zip(self.agents, agent_states):
                action, agent_info = agent(state_tensor, legal_actions=agent_state)
                actions.append(int(action))
                agent_infos.append(agent_info)
            # TODO: gently enforce legality of actions by giving a negative reward and asking again

            # Environment steps
            states, next_rewards, done, info = self.env.step(actions)
            next_game_state = states[0].copy()
            next_agent_states = [state.copy() for state in states[1:]]

            if render:
                self.env.render()

            # Learning
            for agent, action, agent_state, next_agent_state, reward, next_reward, agent_info in zip(
                self.agents, actions, agent_states, next_agent_states, rewards, next_rewards, agent_infos
            ):
                agent.learn(
                    state=state_tensor,
                    legal_actions=agent_state,
                    reward=reward,
                    action=action,
                    done=done,
                    next_state=torch.tensor(next_game_state).to(self.device, self.dtype),
                    next_legal_actions=next_agent_state,
                    next_reward=next_reward,
                    num_episode=self.game,
                    episode_end=done,
                    **agent_info,
                )

            scores += np.array(next_rewards)
            game_state = next_game_state
            agent_states = next_agent_states
            rewards = next_rewards

        self.results.append(scores)
        self.game += 1

    def _set_env_player_names(self):
        names = []
        for agent in self.agents:
            try:
                names.append(agent.__name__)
            except:
                names.append(type(agent).__name__)
        self.env._player_names = names
