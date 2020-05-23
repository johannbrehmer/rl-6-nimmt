from .base import Agent
from .random import DrunkHamster

from .dqn import (
    DQNVanilla,
    DDQNAgent,
    DuellingDQNAgent,
    DuellingDDQNAgent,
    DQN_PRBAgent,
    DDQN_PRBAgent,
    DuellingDDQN_PRBAgent,
    DQN_NStep_Agent,
    D3QN_PRB_NStep,
    Noisy_DQN,
    Noisy_D3QN_PRB_NStep,
    Noisy_D3QN,
)
from .actor_critic import MCActorCriticAgent, NStepActorCriticAgent, ACERAgent
from .policy import ReinforceAgent

DUELLING_DDQN = "duelling_ddqn"

DQN_PRB = "dqn_prb"
DDQN_PRB = "ddqn_prb"
DUELLING_DDQN_PRB = "duelling_ddqn_prb"
DQN_NSTEP = "dqn_nstep"
D3QN_PRB_NSTEP = "d3qn_prb_nstep"
NOISY_DQN = "noisy_dqn"
NOISY_D_QN_PRB_NSTEP = "noisy_d3qn_prb_nstep"
DUELLING_DQN = "duelling_dqn"
DDQN = "ddqn"
DQN = "dqn"
AC_N = "ac-n"
AC_MC = "ac-mc"
ACER = "acer"
REINFORCE = "reinforce"
RANDOM_AGENT = "random"
NOISY_D3QN = "noisy_d3qn"

AGENTS = {
    RANDOM_AGENT: DrunkHamster,
    REINFORCE: ReinforceAgent,
    AC_MC: MCActorCriticAgent,
    AC_N: NStepActorCriticAgent,
    ACER: ACERAgent,
    DQN: DQNVanilla,
    DDQN: DDQNAgent,
    DUELLING_DQN: DuellingDQNAgent,
    DUELLING_DDQN: DuellingDDQNAgent,
    DQN_PRB: DQN_PRBAgent,
    DDQN_PRB: DDQN_PRBAgent,
    DUELLING_DDQN_PRB: DuellingDDQN_PRBAgent,
    DQN_NSTEP: DQN_NStep_Agent,
    D3QN_PRB_NSTEP: D3QN_PRB_NStep,
    NOISY_DQN: Noisy_DQN,
    NOISY_D_QN_PRB_NSTEP: Noisy_D3QN_PRB_NStep,
    NOISY_D3QN: Noisy_D3QN,
}

POLICY_METHODS = [REINFORCE, AC_MC, AC_N, ACER]
DDQN_METHODS = [DDQN, DUELLING_DDQN, DDQN_PRB, DUELLING_DDQN_PRB, NOISY_D_QN_PRB_NSTEP, NOISY_D3QN, D3QN_PRB_NSTEP]
NSTEP_METHODS = [DQN_NSTEP, D3QN_PRB_NSTEP, NOISY_D_QN_PRB_NSTEP]
NOISY_METHODS = [NOISY_DQN, NOISY_D_QN_PRB_NSTEP, NOISY_D3QN, AC_N]
