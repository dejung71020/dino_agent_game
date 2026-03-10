# dino_rl/train.py
import pygame
from env.dino_env import DinoEnv
from agent.dqn_agent import Agent
from render.renderer import Renderer

env = DinoEnv()
agent = Agent()
renderer = Renderer()

agent.load("model.pth") # 이전 학습 데이터 로드

EPISODES = 5000
FAST_MODE = True

for ep in range(EPISODES):
    state = env.reset()
    done = False
    while not done:
        pygame.event.pump()
        action = agent.act(state)
        next_state, reward, done = env.step(action)
        # 메서드 명칭 수정 및 상태 전이 데이터 보존
        agent.memory.push(state, action, reward, next_state, done)
        agent.learn()
        state = next_state

        if not FAST_MODE:
            renderer.draw_game(env, agent, "TRAIN", 0)

    print(f"Episode: {ep} | Score: {env.score} | Total Steps: {agent.steps}")
    
    if ep % 50 == 0:
        agent.save("model.pth")

agent.save("model.pth")