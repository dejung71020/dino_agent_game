# dino_rl/train.py 수정
import pygame
from env.dino_env import DinoEnv
from agent.dqn_agent import Agent
from render.renderer import Renderer
import config

env = DinoEnv()
agent = Agent()
renderer = Renderer()
agent.load("model.pth")

EPISODES = 5000
FAST_MODE = True

for ep in range(EPISODES):
    state = env.reset()
    done = False
    
    while not done:
        # FAST_MODE일 경우 이벤트 루프를 100스텝마다 한 번만 체크하여 오버헤드 감소
        if not FAST_MODE or agent.steps % 100 == 0:
            pygame.event.pump()

        action = agent.act(state)
        next_state, reward, done = env.step(action)
        agent.memory.push(state, action, reward, next_state, done)
        
        # 최적화: 매 스텝이 아닌 특정 주기마다 집중 학습
        if agent.steps % config.LEARN_EVERY == 0:
            for _ in range(config.LEARN_STEPS):
                agent.learn()
        
        state = next_state
        if not FAST_MODE:
            renderer.draw_game(env, agent, "TRAIN", 0)

    # 렌더링 성능에 영향을 주지 않도록 출력 주기 조절
    if ep % 10 == 0:
        print(f"Episode: {ep} | Score: {env.score} | Steps: {agent.steps} | Eps: {agent.epsilon:.3f}")
    
    if ep % 50 == 0:
        agent.save("model.pth")

agent.save("model.pth")