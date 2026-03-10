# dino_rl/agent/replay_buffer.py
import numpy as np
import random

class ReplayBuffer:
    def __init__(self, capacity):
        self.capacity = capacity
        self.buffer = []
        self.priorities = []
        self.pos = 0

    def __len__(self):
        return len(self.buffer)

    def push(self, state, action, reward, next_state, done):
        max_p = max(self.priorities, default=1.0)
        if len(self.buffer) < self.capacity:
            self.buffer.append((state, action, reward, next_state, done))
            self.priorities.append(max_p)
        else:
            self.buffer[self.pos] = (state, action, reward, next_state, done)
            self.priorities[self.pos] = max_p
            self.pos = (self.pos + 1) % self.capacity

    def sample(self, batch_size):
        probs = np.array(self.priorities) ** 0.6
        probs /= probs.sum()
        indices = np.random.choice(len(self.buffer), batch_size, p=probs)
        samples = [self.buffer[i] for i in indices]
        states, actions, rewards, next_states, dones = zip(*samples)
        return (np.array(states), np.array(actions), np.array(rewards), 
                np.array(next_states), np.array(dones))