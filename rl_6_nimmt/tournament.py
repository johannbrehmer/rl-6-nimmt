import torch
import numpy as np
import logging

logger = logging.getLogger(__name__)

from .play import GameSession


class Tournament:
    def __init__(self, min_players=2, max_players=4, baseline_agents=None, baseline_num_games=100, baseline_condition=1000):
        assert 0 < min_players <= max_players

        self.min_players = 2
        self.max_players = 4
        self.baseline_agents = baseline_agents
        self.baseline_num_games = baseline_num_games
        self.baseline_condition = baseline_condition

        self.total_games = 0
        self.agents = dict()
        self.tournament_scores = dict()
        self.tournament_positions = dict()
        self.baseline_scores = dict()
        self.baseline_positions = dict()
        self.played_games = dict()

    def add_player(self, name, agent):
        assert name not in self.agents.keys()

        agent.__name__ = name  # We really shouldn't do this 😬. Also, we really shouldn't use emojis in serious code.

        self.agents[name] = agent
        self.played_games[name] = 0
        self.tournament_scores[name] = []
        self.tournament_positions[name] = []
        self.baseline_scores[name] = []
        self.baseline_positions[name] = []

    def play_game(self, num_players=None):
        if num_players is None:
            num_players = np.random.choice(list(range(self.min_players, self.max_players + 1)), size=1)[0]
        assert len(self) > num_players

        agent_idx = np.random.choice(len(self.agents.keys()), size=num_players, replace=False)
        agent_names = list(self.agents.keys())
        agent_names = [agent_names[i] for i in agent_idx]
        agents = [self.agents[name] for name in agent_names]

        session = GameSession(*agents)
        session.play_game(render=False)
        scores = session.results[0]
        relative_positions = self._compute_relative_positions(scores)

        self.total_games += 1
        for agent_name, score, rel_pos in zip(agent_names, scores, relative_positions):
            self.played_games[agent_name] += 1
            self.tournament_scores[agent_name].append(score)
            self.tournament_positions[agent_name].append(rel_pos)
            if self.played_games[agent_name] % self.baseline_condition == 0:
                self.baseline_eval(agent_name)

    def baseline_eval(self, agent_name):
        if self.baseline_agents is None:
            return

        session = GameSession(self.agents[agent_name], *self.baseline_agents)
        for _ in range(self.baseline_num_games):
            session.play_game(render=False)
        scores = np.mean(np.array(session.results), axis=0)
        relative_positions = self._compute_relative_positions(scores)

        self.baseline_scores[agent_name].append(scores[0])
        self.baseline_positions[agent_name].append(relative_positions[0])

    def __str__(self):
        lines = [f"Tournament after {self.total_games} games:"]
        lines.append("--------------------------------------------------------------------------------------------------")
        lines.append(" Agent                | Games | Tournament score | Tournament pos | Baseline score | Baseline pos ")
        lines.append("--------------------------------------------------------------------------------------------------")

        for name in self.agents.keys():
            t_score = "-" if not self.tournament_scores[name] else f"{np.mean(self.tournament_scores[name]):>5.2f}"
            t_pos = "-" if not self.tournament_positions[name] else f"{np.mean(self.tournament_positions[name]):>5.2f}"
            b_score = "-" if not self.baseline_scores[name] else f"{np.mean(self.baseline_scores[name]):>5.2f}"
            b_pos = "-" if not self.baseline_positions[name] else f"{np.mean(self.baseline_positions[name]):>5.2f}"
            lines.append(f" {name:>20s} | {self.played_games[name]:>5} | {t_score:>16} | {t_pos:>14} | {b_score:>14} | {b_pos:>12} ")

        lines.append("--------------------------------------------------------------------------------------------------")

        return "\n".join(lines)

    @staticmethod
    def _compute_relative_positions(scores):
        epsilon = 0.5
        positions = np.array([np.searchsorted(sorted(scores), score + epsilon) for score in scores], dtype=np.float)
        return (positions - 1) / (len(scores) - 1)

    def __repr__(self):
        return self.__str__()

    def __len__(self):
        return len(self.agents)
