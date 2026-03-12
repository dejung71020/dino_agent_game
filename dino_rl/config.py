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
BATCH_SIZE = 128      # 🚀 기존 64에서 128로 증가 (학습 안정성 극대화)
GAMMA = 0.995         # 🚀 기존 0.99에서 0.995로 증가 (미래 생존 중시)
LR = 0.00025
EPS_START = 1.0
EPS_END = 0.05
EPS_DECAY = 10000     # 탐험률 감소 속도
TARGET_UPDATE_FREQ = 1000
TAU = 0.005 

# 보상(Reward) 및 패널티 시스템
REWARD_SURVIVE = 0.1
REWARD_COIN = 20.0
REWARD_KING_COIN = 50.0
REWARD_PASS = 100.0         # 🚀 기존 40에서 100으로 대폭 상승 (장애물 회피 칭찬)
REWARD_NEAR_MISS = 5.0
REWARD_DEATH = -1000.0      # 🚀 기존 -100에서 -1000으로 하락 (죽음 패널티 극대화)
REWARD_TRAP_DEATH = -1200.0 # 🚀 덫 충돌 시 더 큰 패널티 부여
PENALTY_IDLE_DUCK = -0.2

# 🚀 PER (우선순위 경험 재생) 하이퍼파라미터
PER_ALPHA = 0.6         # 우선순위를 얼마나 강하게 적용할지 (0이면 랜덤, 1이면 오답만)
PER_BETA_START = 0.4    # 중요도 샘플링(IS) 가중치 시작값 (초반 편향을 잡아줌)
PER_BETA_FRAMES = 50000 # 베타 값이 1.0에 도달할 때까지 걸리는 스텝 수

LEARN_EVERY = 4       # 4스텝마다 학습 수행 (환경 실행 속도 향상)
LEARN_STEPS = 2       # 한 번 학습할 때 업데이트 횟수 (GPU/CPU 활용도 극대화)