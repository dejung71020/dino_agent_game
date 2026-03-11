# dino_rl/agent/network.py
import torch
import torch.nn as nn
import torch.nn.functional as F

class DuelingDQN(nn.Module):
    def __init__(self, state_size, action_size):
        super().__init__()
        # 공통 특징 추출층 (🚀 128 -> 256으로 뉴런 수 2배 확장)
        self.fc = nn.Sequential(
            nn.Linear(state_size, 256), nn.ReLU(),
            nn.Linear(256, 256), nn.ReLU()
        )
        # 이득(Advantage) 스트림
        self.adv = nn.Sequential(nn.Linear(256, 128), nn.ReLU(), nn.Linear(128, action_size))
        # 가치(Value) 스트림
        self.val = nn.Sequential(nn.Linear(256, 128), nn.ReLU(), nn.Linear(128, 1))

    def forward(self, x):
        x = self.fc(x)
        adv = self.adv(x)
        val = self.val(x)
        # Dueling 공식 적용: V + (A - mean(A))
        return val + adv - adv.mean(dim=1, keepdim=True)