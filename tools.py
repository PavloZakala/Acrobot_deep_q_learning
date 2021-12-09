import random
from collections import deque

class ReplayMemory(object):

    def __init__(self, capacity):
        self.memory = deque(maxlen=capacity)

    def push(self, state, action, next_state, reward):
        self.memory.append((state, action, next_state, reward))

    def sample(self, batch_size):
        batch_size = min(batch_size, len(self.memory))
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)