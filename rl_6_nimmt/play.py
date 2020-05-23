import torch
import numpy as np
import logging
from .game import SechsNimmtEnv

logger = logging.getLogger(__name__)


class GameSession():
    def __init__(self, *agents, device=torch.device("cpu"), dtype=torch.float):
        """ Initializes a game session, which consists of an arbitrary number of games between the given agents """

        self.device = device
        self.dtype = dtype
        self.agents = [agent.to(self.device, self.dtype) for agent in agents]
        self.num_agents = len(agents)
        self.env = SechsNimmtEnv()

        self.game = 0  # Current game id (= number of finished games)
        self.round = 0  # Round id in current game (= number of finished rounds), where a round means playing through one full stack of cards

        self.results = []  # List of total scores (negative Hornochsen) for each game

    def play_game(self):
        """ Play one game, i.e. until one player hits 66 Hornochsen or whatever it is """

        game_state, agent_states, rewards, done = self.env.reset()
        scores = np.zeros(self.num_agents, dtype=np.int)

        while not done:
            # Agent turns
            actions, agent_infos = [], []
            for agent, agent_state in zip(self.agents, agent_states):
                action, agent_info = agent(game_state, agent_state)
                actions.append(action)
                agent_infos.append(agent_info)

            # Environment steps
            states, next_rewards, done, info = self.env.step(actions)
            next_game_state = states[0]
            next_agent_states = states[1:]

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
