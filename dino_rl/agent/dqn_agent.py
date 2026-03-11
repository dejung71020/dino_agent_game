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
        # 🚀 기존 ReplayBuffer 대신 새롭게 만든 PrioritizedReplayBuffer 적용
        self.memory = PrioritizedReplayBuffer(config.BUFFER_SIZE, alpha=config.PER_ALPHA)
        
        self.steps = 0
        self.epsilon = config.EPS_START

    def act(self, state, train=True):
        if train:
            self.epsilon = config.EPS_END + (config.EPS_START - config.EPS_END) * \
                           np.exp(-1. * self.steps / config.EPS_DECAY)
            if random.random() < self.epsilon:
                return random.randrange(config.ACTION_SIZE)
                
        state_t = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        with torch.no_grad():
            q_values = self.online_net(state_t)
        return q_values.argmax().item()

    def learn(self):
        if len(self.memory) < config.BATCH_SIZE:
            return
            
        # 🚀 베타 값은 학습이 진행될수록 1.0에 가까워지게 설정 (안정성 증가)
        beta = min(1.0, config.PER_BETA_START + self.steps * (1.0 - config.PER_BETA_START) / config.PER_BETA_FRAMES)
        
        # 🚀 데이터를 뽑을 때 인덱스와 가중치도 함께 받아옵니다.
        batch, indices, weights = self.memory.sample(config.BATCH_SIZE, beta)
        states, actions, rewards, next_states, dones = batch
        
        states = torch.FloatTensor(states).to(self.device)
        actions = torch.LongTensor(actions).unsqueeze(1).to(self.device)
        rewards = torch.FloatTensor(rewards).unsqueeze(1).to(self.device)
        next_states = torch.FloatTensor(next_states).to(self.device)
        dones = torch.FloatTensor(dones).unsqueeze(1).to(self.device)
        weights = torch.FloatTensor(weights).unsqueeze(1).to(self.device)

        # 현재 상태의 Q값 계산
        q_values = self.online_net(states).gather(1, actions)
        
        # 다음 상태의 최대 Q값 계산 (Double DQN 방식)
        with torch.no_grad():
            next_actions = self.online_net(next_states).argmax(1).unsqueeze(1)
            next_q_targets = self.target_net(next_states).gather(1, next_actions)
            expected_q_values = rewards + (config.GAMMA * next_q_targets * (1 - dones))

        # 🚀 1. 각 데이터별 오차(TD Error) 계산
        # reduction='none'으로 설정하여 평균을 내지 않고 각각의 오차를 구합니다.
        loss_each = F.mse_loss(q_values, expected_q_values, reduction='none')
        
        # 🚀 2. 버퍼의 우선순위를 업데이트할 수 있도록 넘파이 배열로 변환 (+0.00001은 0 방지용)
        td_errors = loss_each.detach().cpu().numpy().flatten()
        self.memory.update_priorities(indices, td_errors + 1e-5)

        # 🚀 3. 최종 Loss는 각 오차에 IS 가중치를 곱해서 평균을 냅니다.
        loss = (loss_each * weights).mean()

        self.optimizer.zero_grad()
        loss.backward()
        # 그래디언트 클리핑: 너무 큰 오차로 인해 신경망이 파괴되는 것을 방지
        torch.nn.utils.clip_grad_norm_(self.online_net.parameters(), 1.0)
        self.optimizer.step()

        self.steps += 1

        # 타겟 네트워크 소프트 업데이트 (Tau 사용)
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
                print(f"⚠️ 모델 로드 실패 (구조 변경됨). 새롭게 학습합니다.")
                return False
        return False