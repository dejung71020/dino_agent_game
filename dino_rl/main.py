# dino_rl/main.py
import pygame
import os
from env.dino_env import DinoEnv
from agent.dqn_agent import Agent
from render.renderer import Renderer

def load_high_score():
    if os.path.exists("highscore.txt"):
        with open("highscore.txt", "r") as f:
            return int(f.read())
    return 0

def save_high_score(score):
    with open("highscore.txt", "w") as f:
        f.write(str(score))

def main():
    env = DinoEnv()
    agent = Agent()
    renderer = Renderer()
    high_score = load_high_score()
    
    running = True
    mode = "MENU"

    while running:
        if mode == "MENU":
            renderer.draw_menu()
            for event in pygame.event.get():
                if event.type == pygame.QUIT: running = False
                if event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_1: mode = "PLAY"
                    if event.key == pygame.K_2: mode = "TRAIN_VISUAL"
                    if event.key == pygame.K_3: mode = "TRAIN_FAST"
                    if event.key == pygame.K_4: mode = "AI_PLAY"

        elif mode == "TRAIN_FAST":
            print("Fast Training Started...")
            agent.load("model.pth")
            for ep in range(5000):
                state = env.reset()
                done = False
                while not done:
                    pygame.event.pump()
                    action = agent.act(state)
                    state, reward, done = env.step(action)
                    agent.memory.push(state, action, reward, state, done)
                    agent.learn()
                
                if env.score > high_score:
                    high_score = env.score
                    save_high_score(high_score)
                
                print(f"Ep: {ep} | Score: {env.score} | Steps: {agent.steps}")
                if ep % 50 == 0: agent.save("model.pth")
            mode = "MENU"

        else: # PLAY, TRAIN_VISUAL, AI_PLAY
            state = env.reset()
            done = False
            agent.load("model.pth")

            while not done:
                for event in pygame.event.get():
                    if event.type == pygame.QUIT: done = True; running = False
                
                if mode == "PLAY":
                    keys = pygame.key.get_pressed()
                    action = 1 if keys[pygame.K_SPACE] or keys[pygame.K_UP] else (2 if keys[pygame.K_DOWN] else 0)
                elif mode == "TRAIN_VISUAL":
                    action = agent.act(state)
                else: # AI_PLAY
                    action = agent.act(state, train=False)

                next_state, reward, done = env.step(action)
                if mode == "TRAIN_VISUAL":
                    agent.memory.push(state, action, reward, next_state, done)
                    agent.learn()
                
                # 최고 기록 업데이트
                if env.score > high_score:
                    high_score = env.score
                    save_high_score(high_score)
                
                renderer.draw_game(env, agent, mode, high_score)
                state = next_state
            
            if mode == "TRAIN_VISUAL": agent.save("model.pth")
            mode = "MENU"

    pygame.quit()

if __name__ == "__main__":
    main()