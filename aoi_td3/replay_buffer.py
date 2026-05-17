"""Replay buffer for off-policy TD3 updates."""

from collections import deque
import random

import numpy as np
import torch


class ReplayBuffer:
    """Fixed-size transition buffer.

    The storage and sampling behavior matches the original implementation:
    transitions are stored in a deque and sampled uniformly with
    ``random.sample``.
    """

    def __init__(self, max_size):
        self.buffer = deque(maxlen=max_size)

    def add(self, transition):
        self.buffer.append(transition)

    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        state, action, reward, next_state, done = zip(*batch)
        return (
            torch.FloatTensor(np.array(state)),
            torch.FloatTensor(np.array(action)),
            torch.FloatTensor(np.array(reward)).unsqueeze(1),
            torch.FloatTensor(np.array(next_state)),
            torch.FloatTensor(np.array(done)).unsqueeze(1),
        )

    def size(self):
        return len(self.buffer)

    def __len__(self):
        return len(self.buffer)

    def clear(self):
        self.buffer.clear()

    def is_ready(self, batch_size):
        return self.size() >= batch_size

