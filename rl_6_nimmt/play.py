import torch
import numpy as np
import logging

logger = logging.getLogger(__name__)


class GameSession:
    def __init__(self, env, *agents, device=torch.device("cpu"), dtype=torch.float):
        """ Initializes a game session, which consists of an arbitrary number of games between the given agents """

        self.device = device
        self.dtype = dtype
        self.agents = [agent.to(self.device, self.dtype) for agent in agents]
        self.num_agents = len(agents)
        self.env = env
        self.results = []  # List of total scores (negative Hornochsen) for each game
        self.game = 0

        if self.env._player_names is None:
            names = []
            for agent in self.agents:
                try:
                    names.append(agent.__name__)
                except:
                    names.append(type(agent).__name__)
            self.env._player_names = names

    def play_game(self, render=False):
        """ Play one game, i.e. until one player hits 66 Hornochsen or whatever it is """

        states = self.env.reset()
        game_state = states[0]
        agent_states = states[1:]
        done = False
        rewards = np.zeros(self.num_agents, dtype=np.int)
        scores = np.zeros(self.num_agents, dtype=np.int)

        if render:
            self.env.render()

        while not done:
            # Agent turns
            actions, agent_infos = [], []
            for agent, agent_state in zip(self.agents, agent_states):
                action, agent_info = agent(game_state, legal_actions=agent_state)
                actions.append(action)
                agent_infos.append(agent_info)

            # Environment steps
            states, next_rewards, done, info = self.env.step(actions)
            next_game_state = states[0]
            next_agent_states = states[1:]

            if render:
                self.env.render()

            # Learning
            for agent, action, agent_state, next_agent_state, reward, next_reward, agent_info in zip(
                self.agents, actions, agent_states, next_agent_states, rewards, next_rewards, agent_infos
            ):
                agent.learn(
                    game_state=game_state,
                    agent_state=agent_state,
                    reward=reward,
                    action=action,
                    done=done,
                    next_game_state=next_game_state,
                    next_agent_state=next_agent_state,
                    next_reward=next_reward,
                    num_episode=self.game,
                    **agent_info,
                )

            scores += np.array(next_rewards)
            game_state = next_game_state
            agent_states = next_agent_states
            rewards = next_rewards

        self.results.append(scores)
        self.game += 1
