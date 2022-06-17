import random
from collections import deque
import numpy as np

class ReplayBuffer(object):

    def __init__(self, buffer_size) -> None:
        self.buffer_size = buffer_size
        self.num_experiences = 0
        self.buffer = deque()

    def get_batch(self, batch_size):

        experiences = random.sample(self.buffer, batch_size)

        return {
            "states0": np.asarray([exp[0] for exp in experiences], np.float32),
            "actions": np.asarray([exp[1] for exp in experiences], np.float32),
            "rewards": np.asarray([exp[2] for exp in experiences], np.float32),
            "states1": np.asarray([exp[3] for exp in experiences], np.float32),
            "terminals1": np.asarray([exp[4] for exp in experiences], np.float32)
        }

    def add(self, state, action, reward, new_state, done):
        exprience = (state, action, reward, new_state, done)

        if self.num_experiences < self.buffer_size:
            self.buffer.append(exprience)
            self.num_experiences += 1
        else:
            self.buffer.popleft()
            self.buffer.append(exprience)

    def clear(self):
        self.buffer = deque()
        self.num_experiences = 0