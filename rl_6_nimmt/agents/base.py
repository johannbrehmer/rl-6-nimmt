import torch
from torch import nn
from ..utils.replay_buffer import History


class Agent(nn.Module):
    """ Abstract base agent class """

    def __init__(self, env, gamma=0.99, optim_kwargs=None, history_length=None, dtype=torch.float, device=torch.device("cpu")):

        self.gamma = gamma
        self.device = device
        self.dtype = dtype
        self.action_space = env.action_space
        self.num_actions = self.action_space.n
        self.state_length = env.observation_space.shape[0]
        self._init_replay_buffer(history_length)
        self.optimizer = None
        self.optim_kwargs = optim_kwargs
        self.gamma = gamma

        super().__init__()

    def _init_replay_buffer(self, history_length):
        self.history = History(max_length=history_length, dtype=self.dtype, device=self.device)

    def train(self, mode=True):
        super().train(mode=mode)
        if mode:
            optim_kwargs = {} if self.optim_kwargs is None else self.optim_kwargs
            self.optimizer = torch.optim.Adam(params=self.parameters(), **optim_kwargs)

    def forward(self, state, legal_actions, *args, **kwargs):
        """
        Given an environment state, pick the next action and return it.

        Parameters
        ----------
        state : Tensor
            Observed state s_t.

        legal_actions : list of int
            Available actions

        Returns
        -------
        action : int
            Chosen action a_t.

        agent_info : dict
            Additional agent output, depending on the algorithm

        """
        raise NotImplementedError

    def learn(self, state, reward, action, done, next_state, next_reward, episode_end, num_episode, *args, **kwargs):
        """
        Is called at the end of each step, gives the agent the chance to a) update the replay buffer and b) learn its weights.
        """
        raise NotImplementedError
