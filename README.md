# Reinforcement learning for 6 nimmt!

Beating the classic card game 6 nimmt! with reinforcement learning


## Summary


## Getting Started

### Prerequisites

To get things started have a working anaconda or miniconda installation. 


### Installation

For easy use of the git repo, we recommend using ssh keys. Clone the repository with

```
git clone git@github.com:johannbrehmer/reinforcing-fun.git
```

followed by

```
conda env create -f environment.yml
conda activate rl
```


### Let the agents play


### Play yourself



## What we have so far


### Implemented algorithms

 * **Value-based methods:**
    * [x] [DQN](reinforcing_fun/agents/dqn.py) 
    * [x] [DDQN](reinforcing_fun/agents/dqn.py) 
    * [x] [Duelling DDQN](reinforcing_fun/agents/dqn.py) 
    * [x] [Duelling DDQN with Priority Replay Memory](reinforcing_fun/agents/dqn.py)  (and all combinations with the above)
    * [ ] DQN + Noisy Gradients
    * [ ] Rainbow
 * **Pure policy methods:**
    * [x] [REINFORCE](reinforcing_fun/training/policy.py)
    * [ ] Off-policy policy gradient
    * [ ] Trust-region policy optimization
    * [ ] PPO
 * **Actor-critic methods:**
    * [x] [Actor-critic based on MC returns](reinforcing_fun/agents/actor_critic.py)
    * [x] [Actor-critic based on n-step returns](reinforcing_fun/agents/actor_critic.py)
    * [ ] A2C
    * [ ] A3C
    * [ ] IMPALA
    * [x] [ACER](reinforcing_fun/agents/actor_critic.py): work in progress
    * [ ] SAC
 * **Model-based methods:**
    * [ ] ...


### Repository structure

 * [rl-6-nimmt/](rl-6-nimmt/): Implementations of the game and agents
    * [agents/](rl-6-nimmt/agents): Agent models. Each agent should inherit from `agents.base.Agent`.
    * [game/](rl-6-nimmt/game): The game environment
    * [play.py](rl-6-nimmt/play.py): Master function to play a series of 6 nimmt games between different agents.
 * [experiments/](experiments): Experiment data.
 * [environment.yml](environment.yml): conda environment description. Create your own environment from this file with
 `conda env create -f environment.yml`.
 * [black.sh](black.sh): Enforces black code style in all Python files.


## Contributing

### What to do?

Want to implement a new algorithm? Yay! Best check with everyone else on the Slack channel to avoid duplicate efforts.


### Branch policy

We work on personal branches like `marcel` or `jenny` and open PRs to `master` when something is done.


## Credits

### Contributors

- Johann Brehmer
- Marcel Gutsche


### License

This project is licensed under the MIT License - see the [LICENSE.md](LICENSE.md) file for details.


### Acknowledgments

