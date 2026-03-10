import torch
import pygame

from env.dino_env import DinoEnv
from agent.network import DQN
from render.renderer import Renderer
import config


env = DinoEnv()
renderer = Renderer()

model = DQN(config.STATE_SIZE,config.ACTION_SIZE)

model.load_state_dict(torch.load("model.pth"))

model.eval()

state = env.reset()

while True:

    pygame.event.pump()

    state_t = torch.FloatTensor(state).unsqueeze(0)

    action = model(state_t).argmax().item()

    state,reward,done = env.step(action)

    renderer.draw(env)

    if done:
        state = env.reset()