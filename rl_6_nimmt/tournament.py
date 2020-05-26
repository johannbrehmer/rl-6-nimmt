import torch
import numpy as np
import logging

logger = logging.getLogger(__name__)

from .play import GameSession


class Tournament:
    def __init__(self, min_players=2, max_players=4, baseline_agents=None, baseline_num_games=1, baseline_condition=10):
        assert 0 < min_players <= max_players

        self.min_players = min_players
        self.max_players = max_players
        self.baseline_agents = baseline_agents
        self.baseline_num_games = baseline_num_games
        self.baseline_condition = baseline_condition

        self.total_games = 0
        self.agents = dict()
        self.played_games = dict()
        self.tournament_scores = dict()
        self.tournament_positions = dict()
        self.tournament_wins = dict()
        self.baseline_scores = dict()
        self.baseline_positions = dict()
        self.baseline_wins = dict()

    def add_player(self, name, agent):
        assert name not in self.agents.keys()

        agent.__name__ = name  # We really shouldn't do this 😬. Also, we really shouldn't use emojis in serious code.

        self.agents[name] = agent
        self.played_games[name] = 0
        self.tournament_scores[name] = []
        self.tournament_positions[name] = []
        self.tournament_wins[name] = []
        self.baseline_scores[name] = []
        self.baseline_positions[name] = []
        self.baseline_wins[name] = []

    def play_game(self, num_players=None):
        if num_players is None:
            num_players = np.random.choice(list(range(self.min_players, self.max_players + 1)), size=1)[0]
        assert len(self) >= num_players

        agent_idx = np.random.choice(len(self.agents.keys()), size=num_players, replace=False)
        agent_names = list(self.agents.keys())
        agent_names = [agent_names[i] for i in agent_idx]
        agents = [self.agents[name] for name in agent_names]

        session = GameSession(*agents)
        session.play_game(render=False)
        scores = session.results[0]
        relative_positions = self._compute_relative_positions(scores)
        winner = agent_names[np.argmax(scores)]

        self.total_games += 1
        for agent_name, score, rel_pos in zip(agent_names, scores, relative_positions):
            self.played_games[agent_name] += 1
            self.tournament_scores[agent_name].append(score)
            self.tournament_positions[agent_name].append(rel_pos)
            self.tournament_wins[agent_name].append(1.0 if winner == agent_name else 0.0)

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
        winner = float(np.argmax(scores) == 0)

        self.baseline_scores[agent_name].append(scores[0])
        self.baseline_positions[agent_name].append(relative_positions[0])
        self.baseline_wins[agent_name].append(winner)

    def winner(self):
        best = -float("inf")
        winner = None

        for agent_name, agent in self.agents.items():
            if np.mean(self.tournament_positions[agent_name]) > best:
                best = np.mean(self.tournament_positions[agent_name])
                winner = agent

        return winner

    def __str__(self):
        lines = [f"Tournament after {self.total_games} games:"]
        lines.append("----------------------------------------------------------------------------------------------------")
        lines.append(" Agent                | Games | Tournament score | Tournament wins | Baseline score | Baseline wins ")
        lines.append("----------------------------------------------------------------------------------------------------")

        for name in self.agents.keys():
            t_score = "-" if not self.tournament_scores[name] else f"{np.mean(self.tournament_scores[name]):>5.2f}"
            t_pos = "-" if not self.tournament_wins[name] else f"{np.mean(self.tournament_wins[name]):>5.2f}"
            b_score = "-" if not self.baseline_scores[name] else f"{np.mean(self.baseline_scores[name]):>5.2f}"
            b_pos = "-" if not self.baseline_wins[name] else f"{np.mean(self.baseline_wins[name]):>5.2f}"
            lines.append(f" {name:>20s} | {self.played_games[name]:>5} | {t_score:>16} | {t_pos:>15} | {b_score:>14} | {b_pos:>13} ")

        lines.append("----------------------------------------------------------------------------------------------------")

        return "\n".join(lines)

    @staticmethod
    def _compute_relative_positions(scores):
        epsilon = 0.5
        positions_l = np.array([np.searchsorted(sorted(scores), score + epsilon) for score in scores], dtype=np.float)
        positions_r = 1.0 + np.array([np.searchsorted(sorted(scores), score - epsilon) for score in scores], dtype=np.float)
        positions = 0.5 * (positions_l + positions_r)
        return (positions - 1) / (len(scores) - 1)

    def __repr__(self):
        return self.__str__()

    def __len__(self):
        return len(self.agents)
