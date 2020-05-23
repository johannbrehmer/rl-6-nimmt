import random
import numba
import numpy as np
import torch
from collections import defaultdict

import random
import logging

logger = logging.getLogger(__name__)

# Taken and modified from https://github.com/MorvanZhou/Reinforcement-learning-with-tensorflow/blob/master/contents/5.2_Prioritized_Replay_DQN/RL_brain.py


@numba.jit(nopython=True)
def _update(tree, tree_index, priority, total_priority):
    # Change = new priority score - former priority score
    change = priority - tree[tree_index]
    tree[tree_index] = priority

    # then propagate the change through tree
    # this method is faster than the recursive loop
    while tree_index != 0:
        tree_index = (tree_index - 1) // 2
        tree[tree_index] += change
    # assert total_priority > 0
    return tree


@numba.jit(nopython=True)
def _get_leaf_ids(n, tree, priority_segment):
    idx = np.empty((n,), dtype=np.uint32)
    for i in range(n):
        # A value is uniformly sample from each range
        value = priority_segment * i + random.random() * priority_segment
        # print("value:", value)
        # Experience index that correspond to each value is retrieved
        idx[i] = _get_leaf_index(tree, value)
    return idx


@numba.jit(nopython=True)
def _get_leaf_index(tree, v):
    parent_index = 0

    while True:
        left_child_index = 2 * parent_index + 1
        right_child_index = left_child_index + 1

        # If we reach bottom, end the search
        if left_child_index >= len(tree):
            leaf_index = parent_index
            break
        else:  # downward search, always search for a higher priority node
            if v <= tree[left_child_index]:
                parent_index = left_child_index
            else:
                v -= tree[left_child_index]
                parent_index = right_child_index
    return leaf_index


class SumTree:

    # Here we initialize the tree with all nodes = 0, and initialize the data with all values = 0
    def __init__(self, capacity):
        self.data_pointer = 0
        # Number of leaf nodes (final nodes) that contains experiences
        self.capacity = capacity
        self.num_items = 0
        # Generate the tree with all nodes values = 0
        # To understand this calculation (2 * capacity - 1) look at the schema below
        # Remember we are in a binary node (each node has max 2 children) so 2x size of leaf (capacity) - 1 (root node)
        # Parent nodes = capacity - 1
        # Leaf nodes = capacity
        self.tree = np.zeros(2 * capacity - 1)

        # Contains the experiences (so the size of data is capacity)
        self.data = np.zeros(capacity, dtype=object)

    def add(self, priority, data):
        # Look at what index we want to put the experience
        tree_index = self.data_pointer + self.capacity - 1

        """ tree:
                    0
                   / \
                  0   0
                 / \ / \
        tree_index  0 0  0  We fill the leaves from left to right
        """

        # Update data frame
        self.data[self.data_pointer] = data

        # Update the leaf
        self.update(tree_index, priority)

        # Add 1 to data_pointer
        self.data_pointer += 1

        if self.data_pointer >= self.capacity:  # If we're above the capacity, we go back to first index (we overwrite)
            self.data_pointer = 0
        else:
            self.num_items += 1

    def update(self, tree_index, priority):
        self.tree = _update(self.tree, tree_index, priority, self.total_priority)

    # @numba.jit(nopython=True)
    def get_leaf(self, v):
        leaf_index = _get_leaf_index(self.tree, v)

        data_index = leaf_index - self.capacity + 1
        # assert isinstance(self.data[data_index], dict)
        return leaf_index, self.tree[leaf_index], self.data[data_index]

    @property
    def total_priority(self):
        return self.tree[0]  # Returns the root node


class PriorityReplayBuffer:
    """
    A lot is going on here, which needs some explaining:
        1. We want to use priority replay to draw more often from memories/transitions, which have a higher proportion of
        information.
        2. Memories are weighted according to the temporal difference error. Naively implementing this would be inefficient
        (e.g. sorting the array by weights for example) -> SumTree helps here
        3. Due to the weights introduced, we actually contradict our first reason to introduce a random replay buffer
        decorrelation of memories. To avoid this, we borrow an idea from importance sampling.
        4. When calculating the error between q targets and predicted q values, we assign the memories with a high
        priority/high temporal difference error a lower weight. The rationale behind this: "Hey you will see this values quite often,
        so do not overemphasis it too much.
    """

    def __init__(self, max_length=None, dtype=torch.float, device=torch.device("cpu")):
        # Making the tree
        self.dtype = dtype
        self.device = device
        if max_length is None:
            raise ValueError("PriorityReplayBuffer needs max length!")
        self.tree = SumTree(max_length)
        self.absolute_error_upper = 1.0  # clipped abs error
        # stored as ( state, action, reward, next_state ) in SumTree
        self.epsilon = 0.01  # Hyperparameter that we use to avoid some experiences to have 0 probability of being taken
        self.alpha = 0.6  # Hyperparameter that we use to make a tradeoff between taking only exp with high priority and sampling randomly
        self.beta = 0.4  # importance-sampling, from initial value increasing to 1
        self.beta_increment = 0.001

    def store(self, **experience):
        # Find the max priority
        max_priority = np.max(self.tree.tree[-self.tree.capacity :])
        # If the max priority = 0 we can't put priority = 0 since this experience will never have a chance to be selected
        # So we use a minimum priority
        if max_priority == 0:
            max_priority = self.absolute_error_upper

        self.tree.add(max_priority, experience)  # set the max priority for new priority

    def sample(self, n):
        priority_segment = self.tree.total_priority / n  # priority segment
        self.beta = np.min([1.0, self.beta + self.beta_increment])
        start_idx = len(self.tree.tree) - self.tree.capacity
        end_idx = start_idx + self.tree.num_items
        min_prob = np.min(self.tree.tree[start_idx:end_idx]) / self.tree.total_priority  # for later calculate ISweight
        minibatch, b_idx, importance_smapling_weights = self.get_samples(min_prob, n, priority_segment)
        # for key, value in minibatch.items(): # convert to arrays
        #     value = self.stackify(value)
        #     minibatch[key] = value
        return b_idx, importance_smapling_weights, minibatch

    def get_samples(self, min_prob, n, priority_segment):

        leaf_idx = _get_leaf_ids(n, self.tree.tree, priority_segment)
        data_idx = leaf_idx - self.tree.capacity + 1
        priorities = self.tree.tree[leaf_idx]
        data_batch = self.tree.data[data_idx]
        assert not 0 in data_batch, "Wrong data in sample detected"
        probs = priorities / self.tree.total_priority
        importance_smapling_weights = np.power(probs / min_prob, -self.beta)
        # assert isinstance(self.data[data_index], dict)
        minibatch = {k: [dic[k] for dic in data_batch] for k in data_batch[0]}
        # for x in data_batch:
        #      for key, value in x.items():
        #         minibatch[key].append(value)
        return minibatch, leaf_idx, importance_smapling_weights

    def batch_update(self, tree_idx, abs_errors):
        """'
        must be called to update priorities
        """
        abs_errors += self.epsilon  # convert to abs and avoid 0
        if isinstance(abs_errors, torch.Tensor):
            abs_errors = abs_errors.cpu().numpy()
        clipped_errors = np.minimum(abs_errors, self.absolute_error_upper)

        ps = clipped_errors ** self.alpha

        for ti, p in zip(tree_idx, ps):
            self.tree.update(ti, p)

    def __len__(self):
        return self.tree.num_items


class History:
    """ Generic replay buffer. Can accommodate arbitrary fields. """

    def __init__(self, max_length=None, dtype=torch.float, device=torch.device("cpu")):
        self.memories = None
        self.max_length = max_length
        self.data_pointer = 0
        self.is_full = False
        if max_length:
            self.memories = np.empty((max_length,), dtype=object)
        else:
            self.memories = np.empty((128,), dtype=object)  # double memory size each time limit is hit
        self.device = device
        self.dtype = dtype

    def store(self, **kwargs):
        self.memories[self.data_pointer] = kwargs
        self.is_full = False
        self.data_pointer += 1
        if self.max_length is not None and self.data_pointer >= self.max_length:
            self.data_pointer = 0
            self.is_full = True
        if self.data_pointer >= self.memories.shape[0] and self.max_length is None:
            # self.memories.resize(self.memories.shape * 2)  # Raises some ValueError
            self.memories = np.resize(self.memories, self.memories.shape[0] * 2)

    # @timeit
    def sample(self, n):
        idx = random.sample(range(len(self)), k=n)
        data_batch = self.memories[idx]
        minibatch = {k: [dic[k] for dic in data_batch] for k in data_batch[0]}

        return idx, None, minibatch

    def rollout(self, n=None):
        """ When n is not None, returns only the last n entries """
        data_batch = self.memories[: len(self)] if n is None else self.memories[len(self) - n : len(self)]
        minibatch = {k: [dic[k] for dic in data_batch] for k in data_batch[0]}
        return minibatch

    def __len__(self):
        if self.max_length is None:
            return self.data_pointer
        else:
            if self.is_full:
                return self.max_length
            else:
                return self.data_pointer

    def clear(self):
        if self.max_length:
            self.memories = np.empty((self.max_length,), dtype=object)
        else:
            self.memories = np.empty((128,), dtype=object)  # double memory size each time limit is hit
        self.data_pointer = 0

    def __add__(self, other):
        raise DeprecationWarning("Is not used anymore... I hope?")
        assert list(self.memories.keys()) == list(other.memories.keys())

        history = History(self.max_length)
        history.memories = dict()
        for key, val in self.memories.items():
            history.memories[key] = val + other.memories[key]

        return history


class SequentialHistory(History):
    """ Generic replay buffer where each entry represents a sequence of events. Can accommodate arbitrary fields. """

    def __init__(self, max_length=None, dtype=torch.float, device=torch.device("cpu")):
        super().__init__(max_length=max_length, dtype=dtype, device=device)
        self.current_sequence = dict()

    def current_sequence_length(self):
        if len(self.current_sequence) == 0:
            return 0
        else:
            return len(self.current_sequence[list(self.current_sequence.keys())[0]])

    def store(self, **kwargs):
        # Store in temporary sequence buffer
        if self.current_sequence_length() == 0:  # Nothing saved in current sequence
            for key, val in kwargs.items():
                self.current_sequence[key] = [val]
            self.current_sequence["first"] = [True]
        else:
            for key, val in kwargs.items():
                self.current_sequence[key].append(val)
            self.current_sequence["first"].append(False)

    def flush(self):
        """ Push current sequence to ("long-term") memory """
        assert self.current_sequence_length() > 0
        super().store(**self.current_sequence)
        self.current_sequence = dict()
