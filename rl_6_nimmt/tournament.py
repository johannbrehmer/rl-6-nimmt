import torch
import numpy as np
import logging
import copy
import multi_elo

logger = logging.getLogger(__name__)

from .play import GameSession


class Tournament:
    def __init__(self, min_players=2, max_players=4, baseline_agents=None, baseline_num_games=1, baseline_condition=10, elo_initial=1600, elo_k=32):
        assert 0 < min_players <= max_players

        self.min_players = min_players
        self.max_players = max_players
        self.baseline_agents = baseline_agents
        self.baseline_num_games = baseline_num_games
        self.baseline_condition = baseline_condition
        self.elo_initial=elo_initial
        self.elo_k = elo_k

        self.total_games = 0
        self.agents = dict()
        self.descendants = dict()
        self.active = dict()
        self.played_games = dict()
        self.tournament_scores = dict()
        self.tournament_positions = dict()
        self.tournament_wins = dict()
        self.baseline_scores = dict()
        self.baseline_positions = dict()
        self.baseline_wins = dict()
        self.elos = dict()

    def add_player(self, name, agent):
        assert name not in self.agents.keys()

        agent.__name__ = name  # We really shouldn't do this 😬. Also, we really shouldn't use emojis in serious code.

        self.agents[name] = agent
        self.descendants[name] = name
        self.active[name] = True
        self.played_games[name] = 0
        self.tournament_scores[name] = []
        self.tournament_positions[name] = []
        self.tournament_wins[name] = []
        self.baseline_scores[name] = []
        self.baseline_positions[name] = []
        self.baseline_wins[name] = []
        self.elos[name] = [self.elo_initial]

    def copy_player(self, old_name, new_name):
        for q in [self.active, self.elos, self.descendants, self.played_games, self.tournament_scores, self.tournament_positions, self.tournament_wins, self.baseline_scores, self.baseline_positions, self.baseline_wins]:
            q[new_name] = copy.deepcopy(q[old_name])

        # Deepcopy does not work for PyTorch models, so this is ugly
        torch.save(self.agents[old_name], "temp_model.pt")
        self.agents[new_name] = torch.load("temp_model.pt")

    def remove_player(self, name, full_delete=False):
        if full_delete:
            del self.agents[name]
            del self.active[name]
            del self.descendants[name]
            del self.played_games[name]
            del self.tournament_scores[name]
            del self.tournament_positions[name]
            del self.tournament_wins[name]
            del self.baseline_scores[name]
            del self.baseline_positions[name]
            del self.baseline_wins[name]
            del self.elos[name]
        else:
            self.active[name] = False

    def evolve(self, copies=(2,), max_players=None, max_per_descendant=2, metric="elo"):
        if metric == "tournament_scores":
            scores = self.tournament_scores
            reverse = True
            mean = True
        elif metric == "tournament_positions":
            scores = self.tournament_positions
            reverse = False
            mean = True
        elif metric == "tournament_wins":
            scores = self.tournament_wins
            reverse = False
            mean = True
        elif metric == "elo":
            scores = self.elos
            reverse = True
            mean = False
        else:
            raise NotImplementedError(metric)

        if mean:
            key = lambda x : (np.mean(scores[x]) if len(scores[x]) > 0 else 0.)
        else:
            key = lambda x : (scores[x][-1] if len(scores[x]) > 0 else 0.)

        agent_names = self.active_agents()
        agents_by_scores = sorted(agent_names, key=key, reverse=reverse)

        new_count = 0
        new_descendants = {}
        for pos, name in enumerate(agents_by_scores):
            desc = self.descendants[name]
            if self.descendants[name] not in new_descendants:
                new_descendants[desc] = 0

            if pos < len(copies):
                copy = copies[pos]
                logger.info(f"Copying player {name} into {copy} instances!")
            elif max_players is not None and new_count >= max_players:
                copy = 0
                logger.info(f"Removing player {name}")
            elif max_per_descendant is not None and new_descendants[desc] >= max_per_descendant:
                copy = 0
                logger.info(f"Removing player {name}")
            else:
                copy = 1

            for c in range(copy):
                self.copy_player(name, f"{name}_{c}")
            self.remove_player(name, full_delete=copy>0)

            new_count += copy
            new_descendants[desc] += copy

    def play_game(self, num_players=None):
        agent_names, agents = self._choose_players(num_players)

        session = GameSession(*agents)
        session.play_game(render=False)
        scores = session.results[0]
        self.score_game(agent_names, scores)

    def score_game(self, agent_names, scores):
        relative_positions = self._compute_relative_positions(scores)
        winner = agent_names[np.argmax(scores)]
        new_elos = self._compute_elos(agent_names, scores)

        self.total_games += 1

        for agent_name, score, rel_pos, elo in zip(agent_names, scores, relative_positions, new_elos):
            self.played_games[agent_name] += 1
            self.tournament_scores[agent_name].append(score)
            self.tournament_positions[agent_name].append(rel_pos)
            self.tournament_wins[agent_name].append(1.0 if winner == agent_name else 0.0)
            self.elos[agent_name].append(elo)

            if self.played_games[agent_name] % self.baseline_condition == 0:
                self.baseline_eval(agent_name)

    def _compute_elos(self, agent_names, scores):
        old_elos = [self.elos[name][-1] for name in agent_names]
        positions = self._compute_absolute_positions(scores)
        players = [multi_elo.EloPlayer(place=place, elo=old_elo) for place, old_elo in zip(positions, old_elos)]

        new_elos = multi_elo.calc_elo(players, self.elo_k)

        return new_elos

    def _choose_players(self, num_players):
        if num_players is None:
            num_players = np.random.choice(list(range(self.min_players, self.max_players + 1)), size=1)[0]

        assert len(self) >= num_players

        agent_names = self.active_agents()
        agent_idx = np.random.choice(len(agent_names), size=num_players, replace=False)
        agent_names = [agent_names[i] for i in agent_idx]
        agents = [self.agents[name] for name in agent_names]

        return agent_names, agents

    def active_agents(self):
        return [name for name in self.agents.keys() if self.active[name]]

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
        hline = "-----------------------------------------------------------------"
        lines = [f"Tournament after {self.total_games} games:"]
        lines.append(hline)
        lines.append(" Agent                | Games | Mean score | Win fraction |  ELO ")
        lines.append(hline)

        for name in self.agents.keys():
            if not self.active[name]:
                continue

            score = "-" if not self.tournament_scores[name] else f"{np.mean(self.tournament_scores[name]):>5.2f}"
            wins = "-" if not self.tournament_wins[name] else f"{np.mean(self.tournament_wins[name]):>5.2f}"

            lines.append(f" {name:>20s} | {self.played_games[name]:>5} | {score:>10} | {wins:>12} | {self.elos[name][-1]:>4.0f} ")

        lines.append(hline)

        for name in self.agents.keys():
            if self.active[name]:
                continue

            score = "-" if not self.tournament_scores[name] else f"{np.mean(self.tournament_scores[name]):>5.2f}"
            wins = "-" if not self.tournament_wins[name] else f"{np.mean(self.tournament_wins[name]):>5.2f}"

            lines.append(f" {name:>20s} | {self.played_games[name]:>5} | {score:>10} | {wins:>12} | {self.elos[name][-1]:>4.0f} ")

        if lines[-1] != hline:
            lines.append(hline)

        return "\n".join(lines)

    @staticmethod
    def _compute_absolute_positions(scores):
        """ 0 = best, n_players - 1 = worst """
        epsilon = 0.5
        positions_l = np.array([np.searchsorted(sorted(- scores), - score - epsilon) for score in scores], dtype=np.float)
        positions_r = 1.0 + np.array([np.searchsorted(sorted(- scores), - score + epsilon) for score in scores], dtype=np.float)
        positions = 0.5 * (positions_l + positions_r)
        return positions

    @staticmethod
    def _compute_relative_positions(scores):
        """ 1 = best, 0 = worst """
        epsilon = 0.5
        positions_l = np.array([np.searchsorted(sorted(scores), score + epsilon) for score in scores], dtype=np.float)
        positions_r = 1.0 + np.array([np.searchsorted(sorted(scores), score - epsilon) for score in scores], dtype=np.float)
        positions = 0.5 * (positions_l + positions_r)
        return (positions - 1) / (len(scores) - 1)

    def __repr__(self):
        return self.__str__()

    def __len__(self):
        return len(self.active_agents())
