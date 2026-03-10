# dino_rl/play_ai.py
import pygame
import config
from env.dino_env import DinoEnv
from agent.dqn_agent import Agent
from render.renderer import Renderer

env = DinoEnv()
renderer = Renderer()
agent = Agent()

# 통일된 로드 방식 사용
if not agent.load("model.pth"):
    print("학습된 모델(model.pth)이 없습니다.")
    exit()

state = env.reset()

while True:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            pygame.quit()
            exit()

    # AI 플레이 시에는 탐험(Random action)을 배제
    action = agent.act(state, train=False)
    state, reward, done = env.step(action)
    
    renderer.draw_game(env, agent, "AI_PLAY", 0)

    if done:
        state = env.reset()
        pygame.time.delay(500)