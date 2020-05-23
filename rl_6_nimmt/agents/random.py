from .base import Agent


class DrunkHamster(Agent):
    """ Agent that makes totally random decisions """

    def forward(self, state, **kwargs):
        """
        Given an environment state, pick the next action randomly. No further information is returned.


        Parameters
        ----------
        state : Tensor
            Observed state s_t

        Returns
        -------
        action : int
            Chosen action a_t.

        """
        action = self.action_space.sample()
        return action, {}

    def learn(self, state, reward, action, done, next_state, next_reward, episode_end, num_episode):
        return 0.0
