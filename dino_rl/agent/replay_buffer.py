# dino_rl/agent/replay_buffer.py
import numpy as np
import random
from collections import deque

class PrioritizedReplayBuffer:
    def __init__(self, capacity, alpha=0.6):
        self.capacity = capacity
        self.alpha = alpha
        self.buffer = []
        self.pos = 0
        # 각 경험의 우선순위를 저장하는 배열
        self.priorities = np.zeros((capacity,), dtype=np.float32)

    def push(self, state, action, reward, next_state, done):
        # 새로운 경험은 무조건 한 번은 학습되도록 가장 높은 우선순위를 부여함
        max_prio = self.priorities.max() if self.buffer else 1.0

        if len(self.buffer) < self.capacity:
            self.buffer.append((state, action, reward, next_state, done))
        else:
            self.buffer[self.pos] = (state, action, reward, next_state, done)

        self.priorities[self.pos] = max_prio
        self.pos = (self.pos + 1) % self.capacity

    def sample(self, batch_size, beta=0.4):
        if len(self.buffer) == self.capacity:
            prios = self.priorities
        else:
            prios = self.priorities[:self.pos]

        # 우선순위에 알파(alpha) 제곱을 하여 확률 분포 생성
        probs = prios ** self.alpha
        probs /= probs.sum()

        # 확률에 따라 뽑을 데이터의 인덱스 선택 (오답일수록 잘 뽑힘)
        indices = np.random.choice(len(self.buffer), batch_size, p=probs)
        samples = [self.buffer[idx] for idx in indices]

        # 편향을 보정하기 위한 중요도(IS) 가중치 계산
        total = len(self.buffer)
        weights = (total * probs[indices]) ** (-beta)
        weights /= weights.max() # 정규화
        weights = np.array(weights, dtype=np.float32)

        states, actions, rewards, next_states, dones = zip(*samples)
        return (np.array(states), np.array(actions), np.array(rewards, dtype=np.float32),
                np.array(next_states), np.array(dones, dtype=np.uint8)), indices, weights

    def update_priorities(self, batch_indices, batch_priorities):
        # AI가 채점한 오답 정도(TD Error)로 기존 우선순위를 업데이트
        for idx, prio in zip(batch_indices, batch_priorities):
            self.priorities[idx] = prio

    def __len__(self):
        return len(self.buffer)