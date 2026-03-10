import pygame
import torch

from env.dino_env import DinoEnv
from agent.dqn_agent import Agent
from render.renderer import Renderer

env = DinoEnv()
agent = Agent()
renderer = Renderer()

EPISODES = 5000
FAST_MODE = False

for ep in range(EPISODES):

    state = env.reset()
    done = False

    while not done:

        pygame.event.pump()

        action = agent.act(state)

        next_state,reward,done = env.step(action)

        agent.remember(state,action,reward,next_state,done)

        agent.learn()

        state = next_state

        if not FAST_MODE:
            renderer.draw(env)

    print("episode",ep,"score",env.score)

torch.save(agent.online_net.state_dict(),"model.pth")