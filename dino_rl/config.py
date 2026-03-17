# dino_rl/config.py
import torch

# 화면 및 프레임 설정
SCREEN_WIDTH = 1000
SCREEN_HEIGHT = 400
FPS = 60

# 게임 물리 법칙
GRAVITY = 0.8
JUMP_VELOCITY = -15
DUCK_GRAVITY = 2.0
START_SPEED = 7
MAX_SPEED = 18
SPEED_INCREMENT = 0.0005

# 강화학습(RL) 하이퍼파라미터
STATE_SIZE = 16
ACTION_SIZE = 3
BUFFER_SIZE = 100000
BATCH_SIZE = 128
GAMMA = 0.995
LR = 0.00025
EPS_START = 1.0
EPS_END = 0.01          # 🚀 최저 탐험률을 0.05 -> 0.01로 낮춰 후반부 안정성 도모
EPS_DECAY = 50000       # 🚀 10000 -> 50000으로 대폭 증가 (고속 구간까지 충분히 탐험)
TARGET_UPDATE_FREQ = 1000
TAU = 0.001             # 🚀 0.005 -> 0.001로 감소 (타겟 네트워크의 급격한 변동 방지)

# 보상(Reward) 및 패널티 시스템 최적화 (밸런스 패치)
REWARD_SURVIVE = 0.1
REWARD_COIN = 10.0
REWARD_KING_COIN = 30.0
REWARD_PASS = 20.0          # 🚀 100 -> 20으로 하향 (점수 뻥튀기 방지, 생존 본연에 집중)
REWARD_NEAR_MISS = 2.0
REWARD_DEATH = -100.0       # 🚀 -1000 -> -100으로 대폭 완화 (과도한 방어적 플레이 억제)
REWARD_TRAP_DEATH = -120.0
PENALTY_IDLE_DUCK = -0.1

# PER (우선순위 경험 재생) 하이퍼파라미터
PER_ALPHA = 0.6
PER_BETA_START = 0.4
PER_BETA_FRAMES = 50000 # EPS_DECAY와 주기를 맞춤

LEARN_EVERY = 4
LEARN_STEPS = 2