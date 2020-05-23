import numpy as np
import sys
import torch
import tqdm
sys.path.append("../")

from reinforcing_fun.utils import PriorityReplayBuffer

def run_priority_replay_memory_test():
    pbr = PriorityReplayBuffer(max_length=100, dtype=torch.double)
    experience = {'a': torch.tensor([20, ]), 'b': torch.tensor([30, ])}
    for k in range(10):
        pbr.store(**experience)
    for k in tqdm.tqdm(range(1000)):
        pbr.store(**experience)
        b_idx, importance_smapling_weights, minibatch = pbr.sample(10)
        assert importance_smapling_weights.mean() != 0
        pbr.batch_update(b_idx, torch.from_numpy(np.random.random_sample((1,))))


if __name__ == "__main__":
    run_priority_replay_memory_test()