# Beating 6 nimmt! with reinforcement learning
 
Beating the classic card game 6 nimmt! with reinforcement learning


### 6 nimmt!

[6 nimmt!](https://boardgamegeek.com/boardgame/432/6-nimmt) is an award-winning card game for two to ten players from 1994. Quoting [Wikipedia](https://en.wikipedia.org/wiki/6_Nimmt!):
> The game has 104 cards, each bearing a number and one to seven bull's heads symbols that represent penalty points. A round of ten turns is played where all players place one card of their choice onto the table. The placed cards are arranged on four rows according to fixed rules. If placed onto a row that already has five cards then the player receives those five cards, which count as penalty points that are totaled up at the end of the round.

6 nimmt! is a competitive game of incomplete information and a large amount of stochasticity. Playing well requires a fair bit of planning. The simultaneous game play lends itself to mind games and bluffing, while some long-term strategy is necessary to avoid ending up in difficult end-game positions.

We implemented a slightly simplified version of 6 nimmt! as an [OpenAI gym](https://gym.openai.com/) environment. Unlike in the original game, when playing a lower card than the last card on all stacks, the player cannot freely choose which stack to replace, but instead will always take the stack with the smallest number of penalty points.


### RL agents

So far we have implemented the following agents:
- **D3QN**: A deep dueling double Q-learning algorithm with priority replay buffer and noisy nets. This has most elements of [Rainbow](https://arxiv.org/abs/1710.02298).
- **REINFORCE**: The original policy gradient algorithm.
- **ACER**: An off-policy actor-critic agent with truncated importance sampling with bias correction. Unlike [the original ACER](https://arxiv.org/abs/1611.01224), our implementation does not use trust region policy optimization or dueling stochastic networks.
- **MCS**: A Monte-Carlo search in which all actions are sampled uniformly.
- **Alpha0.5**: Policy-guided Monte-Carlo tree search based on PUCT bounds. Inspired by [AlphaZero](https://arxiv.org/abs/1712.01815) and similar algorithms, modified for the incomplete information scenario of 6 nimmt!.
- **Random**: A random policy baseline.
- **Human**: A human player. 


### Results

As a first test, we ran a simple self-play tournament. Starting with five untrained agents, we played 4000 games in total. For each game we randomly selected two, three, or four agents to play (and learn). Every 400 games we cloned the best-performing agent and kicked out some of the poorer-performing ones.

Results over all games:


| Agent | Games played | Mean score | Win fraction | ELO |
|---|---|---|---|---|
| **Alpha0.5** |  1932 |      -7.92 |         0.41 | 1764 |
| **MCS** |  1970 |      -8.26 |         0.39 | 1827 |
| **ACER** |  1032 |     -12.50 |         0.18 | 1639 |
| **Random** |  1057 |     -13.42 |         0.21 | 1572 |
| **D3QN** |   815 |     -13.19 |         0.18 | 1494 |

This is how the performance (measured in ELO) of the models developed during the course of the tournament:



Clearly the model-based methods perform much better than the model-free ones, which struggle to even clearly beat the random baseline. Due to the stochastic nature of the game, the win probabilities and ELO differences are not nearly as drastic as, say, for chess. Note that we haven't optimized any hyperparameters yet.

Finally, we conducted a best-of-5 game between the Alpha0.5 agent trained in this tournament and Merle, one of the best 6 nimmt! players in our group of friends. These are the scores:

| Game | 1 | 2 | 3 | 4 | 5 |
|---|---|---|---|---|---|
| **Merle** | | | | | |
| **Alpha0.5** | | | | | |

!




### Running the code

To get things started, have a working anaconda or miniconda installation. Clone the repository with

```
git clone git@github.com:johannbrehmer/rl-6nimmt.git
```

followed by

```
conda env create -f environment.yml
conda activate rl
```

Both agent self-play and games between a human player and trained agents are demonstrated in [simple_tournament.ipynb](experiments/simple_tournament.ipynb).


## Contributors

This was put together by Johann Brehmer and Marcel Gutsche.
