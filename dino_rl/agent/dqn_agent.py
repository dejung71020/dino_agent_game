# dino_rl/agent/dqn_agent.py
import torch
import torch.optim as optim
import numpy as np
import random
import config
import os
from agent.network import DuelingDQN
from agent.replay_buffer import ReplayBuffer

class Agent:
    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.online_net = DuelingDQN(config.STATE_SIZE, config.ACTION_SIZE).to(self.device)
        self.target_net = DuelingDQN(config.STATE_SIZE, config.ACTION_SIZE).to(self.device)
        self.target_net.load_state_dict(self.online_net.state_dict())
        
        self.optimizer = optim.Adam(self.online_net.parameters(), lr=config.LR)
        self.memory = ReplayBuffer(config.BUFFER_SIZE)
        self.steps = 0

    def act(self, state, train=True):
        # 저장된 steps를 기반으로 Epsilon 계산
        eps = config.EPS_END + (config.EPS_START - config.EPS_END) * \
              np.exp(-1. * self.steps / config.EPS_DECAY) if train else 0.02
        
        if train: self.steps += 1
        
        if random.random() < eps:
            return random.randint(0, config.ACTION_SIZE - 1)
        
        state_t = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        with torch.no_grad():
            return self.online_net(state_t).argmax().item()

    def learn(self):
        if len(self.memory) < config.BATCH_SIZE: return
        s, a, r, s2, d = self.memory.sample(config.BATCH_SIZE)
        
        s = torch.FloatTensor(s).to(self.device)
        a = torch.LongTensor(a).unsqueeze(1).to(self.device)
        r = torch.FloatTensor(r).to(self.device)
        s2 = torch.FloatTensor(s2).to(self.device)
        d = torch.FloatTensor(d).to(self.device)

        curr_q = self.online_net(s).gather(1, a).squeeze()
        next_actions = self.online_net(s2).argmax(1).unsqueeze(1)
        next_q = self.target_net(s2).gather(1, next_actions).squeeze()
        target_q = r + config.GAMMA * next_q * (1 - d)

        loss = torch.nn.functional.mse_loss(curr_q, target_q.detach())
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.online_net.parameters(), 1.0)
        self.optimizer.step()

        for t, o in zip(self.target_net.parameters(), self.online_net.parameters()):
            t.data.copy_(config.TAU * o.data + (1 - config.TAU) * t.data)

    # 가중치와 steps를 함께 저장
    def save(self, path="model.pth"):
        torch.save({
            'model_state_dict': self.online_net.state_dict(),
            'steps': self.steps
        }, path)

    # 가중치와 steps를 함께 불러오기
    def load(self, path="model.pth"):
        if os.path.exists(path):
            checkpoint = torch.load(path, map_location=self.device)
            self.online_net.load_state_dict(checkpoint['model_state_dict'])
            self.target_net.load_state_dict(self.online_net.state_dict())
            self.steps = checkpoint.get('steps', 0)
            return True
        return False