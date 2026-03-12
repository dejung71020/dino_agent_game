# dino_rl/agent/dqn_agent.py
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import random
import os
import config
from .network import DuelingDQN
from .replay_buffer import PrioritizedReplayBuffer

class Agent:
    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        self.online_net = DuelingDQN(config.STATE_SIZE, config.ACTION_SIZE).to(self.device)
        self.target_net = DuelingDQN(config.STATE_SIZE, config.ACTION_SIZE).to(self.device)
        self.target_net.load_state_dict(self.online_net.state_dict())
        self.target_net.eval()
        
        self.optimizer = optim.Adam(self.online_net.parameters(), lr=config.LR)
        self.memory = PrioritizedReplayBuffer(config.BUFFER_SIZE, alpha=config.PER_ALPHA)
        
        self.steps = 0  # 이 값은 이제 '환경 스텝'을 의미합니다.
        self.epsilon = config.EPS_START

    def act(self, state, train=True):
        if train:
            # 🚀 환경 스텝에 기반하여 탐험률 계산
            self.epsilon = config.EPS_END + (config.EPS_START - config.EPS_END) * \
                           np.exp(-1. * self.steps / config.EPS_DECAY)
            
            # 행동을 선택할 때마다 스텝 증가 (학습 루프의 데드락 방지)
            self.steps += 1
            
            if random.random() < self.epsilon:
                return random.randrange(config.ACTION_SIZE)
                
        state_t = torch.as_tensor(state, dtype=torch.float32, device=self.device).unsqueeze(0)
        with torch.no_grad():
            q_values = self.online_net(state_t)
        return q_values.argmax().item()

    def learn(self):
        # 버퍼에 충분한 데이터가 쌓일 때까지 대기
        if len(self.memory) < config.BATCH_SIZE:
            return
            
        beta = min(1.0, config.PER_BETA_START + self.steps * (1.0 - config.PER_BETA_START) / config.PER_BETA_FRAMES)
        batch, indices, weights = self.memory.sample(config.BATCH_SIZE, beta)
        states, actions, rewards, next_states, dones = batch
        
        states = torch.as_tensor(states, device=self.device)
        actions = torch.as_tensor(actions, device=self.device, dtype=torch.long).unsqueeze(1)
        rewards = torch.as_tensor(rewards, device=self.device).unsqueeze(1)
        next_states = torch.as_tensor(next_states, device=self.device)
        dones = torch.as_tensor(dones, device=self.device, dtype=torch.float32).unsqueeze(1)
        weights = torch.as_tensor(weights, device=self.device).unsqueeze(1)

        q_values = self.online_net(states).gather(1, actions)
        
        with torch.no_grad():
            next_actions = self.online_net(next_states).argmax(1).unsqueeze(1)
            next_q_targets = self.target_net(next_states).gather(1, next_actions)
            expected_q_values = rewards + (config.GAMMA * next_q_targets * (1 - dones))

        loss_each = F.mse_loss(q_values, expected_q_values, reduction='none')
        
        td_errors = loss_each.detach().cpu().numpy().flatten()
        self.memory.update_priorities(indices, td_errors + 1e-5)

        loss = (loss_each * weights).mean()

        self.optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.online_net.parameters(), 1.0)
        self.optimizer.step()

        # 💡 타겟 네트워크 소프트 업데이트 (self.steps 증가 로직은 act로 이동함)
        with torch.no_grad():
            for target_param, online_param in zip(self.target_net.parameters(), self.online_net.parameters()):
                target_param.data.copy_(config.TAU * online_param.data + (1.0 - config.TAU) * target_param.data)

    def save(self, path="model.pth"):
        torch.save({
            'model_state_dict': self.online_net.state_dict(),
            'steps': self.steps
        }, path)

    def load(self, path="model.pth"):
        if os.path.exists(path):
            try:
                checkpoint = torch.load(path, map_location=self.device)
                if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
                    self.online_net.load_state_dict(checkpoint['model_state_dict'])
                    self.steps = checkpoint.get('steps', 0)
                else:
                    self.online_net.load_state_dict(checkpoint)
                    self.steps = 0
                self.target_net.load_state_dict(self.online_net.state_dict())
                print("✅ 기존 모델 로드 완료!")
                return True
            except Exception as e:
                print(f"⚠️ 모델 로드 실패. 새롭게 학습합니다.")
                return False
        return False