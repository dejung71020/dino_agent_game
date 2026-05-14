# 🦖 Dino Agent Game — 강화학습 크롬 공룡 게임 봇

> PyTorch와 DQN(Deep Q-Network) 알고리즘을 활용하여 크롬 다이노 게임을 스스로 학습하고 플레이하는 AI 에이전트

---

## 📌 프로젝트 개요

| 항목 | 내용 |
| --- | --- |
| 구분 | 개인 프로젝트 (KDT) |
| 담당 | 강화학습(RL) 환경 구축 · 모델 설계 및 학습 · 게임 로직 구현 |
| 알고리즘 | Deep Q-Network (DQN) |

### 기획 배경 및 목표

* **도전 과제:** 동적으로 변하는 게임 속도와 장애물(선인장, 익룡)의 무작위성을 AI가 스스로 파악하고 최적의 행동을 선택하도록 만듦.
* **해결 방안:** 행동에 대한 보상(Reward)과 페널티(Penalty)를 설계하고, 오프폴리시(Off-policy) 강화학습 알고리즘인 DQN을 도입하여 스스로 생존 시간을 극대화하는 에이전트를 개발.

---

## 🛠 기술 스택

| 분류 | 기술 |
| --- | --- |
| Language | Python 3.x |
| AI / ML | PyTorch, DQN Algorithm |
| Environment | Custom Environment (OpenAI Gym 스타일), Pygame(렌더링) |
| Data | Experience Replay Buffer |

---

## ✨ 핵심 구현

### 1. 커스텀 강화학습 환경 구축 (Custom Environment)

크롬 공룡 게임의 룰을 강화학습에 맞게 상태(State), 행동(Action), 보상(Reward)으로 재설계했습니다. (`dino_env.py`)

* **State (상태):** 공룡의 y좌표, 장애물과의 거리, 장애물의 크기, 현재 게임 속도 등
* **Action (행동):** `[0: Jump, 1: Duck, 2: Do Nothing]`
* **Reward (보상):** 생존 시 `+1`, 장애물 충돌 시 `-10` (충돌 페널티 부여)

### 2. Deep Q-Network (DQN) 아키텍처 구현

PyTorch를 활용하여 Q-Value를 근사하는 신경망을 구축하고 학습 안정성을 높였습니다. (`dqn_agent.py`, `network.py`)

```python
# Replay Buffer를 통한 경험 축적 (데이터 간의 상관관계 제거)
class ReplayBuffer:
    def push(self, state, action, reward, next_state, done):
        ...

# Target Network와 Policy Network 분리
# 일정한 주기마다 Policy Network의 가중치를 Target Network로 복사하여 학습 타겟을 고정

```

* **Replay Buffer:** 과거의 경험(Transition)을 큐(Queue)에 저장하고 무작위로 미니배치를 추출하여 학습.
* **Epsilon-Greedy 탐험:** 초기에는 무작위 행동으로 환경을 탐험(Exploration)하고, 점차 학습된 Q-Value에 따라 행동(Exploitation)하도록 감쇠(Decay) 적용.

### 3. 학습 및 추론 파이프라인 분리

학습 전용 스크립트와 시각화(렌더링)가 포함된 플레이 스크립트를 완벽히 분리했습니다.

* **`train.py`**: 빠른 에피소드 반복을 위해 화면 렌더링을 끄고 백그라운드에서 모델을 고속 학습. 최적의 가중치는 `model.pth`로 저장.
* **`play_ai.py`**: 저장된 가중치(`model.pth`)를 로드하여 AI가 실제로 게임을 플레이하는 시각적 결과물 확인. 최고 점수는 `highscore.txt`에 기록.

---

## 🚀 실행 방법

```bash
# 1. 저장소 클론
git clone https://github.com/dejung71020/dino_agent_game.git
cd dino_agent_game

# 2. 패키지 설치 (PyTorch, Pygame 등 필요)
pip install -r requirements.txt

# 3. AI 플레이 관전하기 (미리 학습된 모델 사용 시)
python dino_rl/play_ai.py

# 4. 처음부터 모델 다시 학습시키기
python dino_rl/train.py

```

---

## 📁 프로젝트 구조

```text
dino_agent_game/
├── dino_rl/
│   ├── agent/
│   │   ├── dqn_agent.py      # DQN 알고리즘 코어 로직
│   │   ├── network.py        # PyTorch 기반 신경망 모델
│   │   └── replay_buffer.py  # 경험 리플레이 버퍼
│   ├── assets/               # 공룡, 선인장, 익룡 등 게임 이미지 리소스
│   ├── env/
│   │   ├── dino_env.py       # 강화학습 환경 (State, Step, Reset)
│   │   └── entities.py       # 게임 객체 물리 엔진 (Dino, Cactus 등)
│   ├── render/
│   │   └── renderer.py       # Pygame 기반 화면 렌더링
│   ├── config.py             # 하이퍼파라미터 및 게임 설정 값
│   ├── main.py               # 직접 플레이 모드 (사람)
│   ├── play_ai.py            # AI 플레이 모드 (추론)
│   ├── train.py              # AI 학습 스크립트
│   └── highscore.txt         # 최고 점수 기록
├── model/
│   └── model.pth             # 학습이 완료된 PyTorch 모델 가중치
└── README.md

```
