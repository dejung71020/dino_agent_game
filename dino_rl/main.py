# dino_rl/main.py
import pygame
import os
import numpy as np
import config
import matplotlib.pyplot as plt
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
                if event.type == pygame.QUIT: 
                    running = False
                if event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_1: mode = "PLAY"
                    if event.key == pygame.K_2: mode = "TRAIN_VISUAL"
                    if event.key == pygame.K_3: mode = "TRAIN_FAST"
                    if event.key == pygame.K_4: mode = "AI_PLAY"
                    if event.key == pygame.K_5: mode = "TRAIN_ALL_IN" # 5번 모드 추가
                    if event.key == pygame.K_ESCAPE: running = False

        elif mode == "TRAIN_ALL_IN":
            print("🔥 최고 속도 학습 시작! (그래프 끄고 콘솔 출력만 진행)")
            agent.load("model.pth")
            stop_training = False
            
            for ep in range(100000): 
                # (중략 - 이벤트 감지 로직은 그대로 유지)
                
                state = env.reset()
                done = False
                step_count = 0
                
                while not done:
                    step_count += 1
                    if step_count % 100 == 0:
                        pygame.event.pump()
                        keys = pygame.key.get_pressed()
                        if keys[pygame.K_ESCAPE]:
                            stop_training = True
                            break

                    action = agent.act(state)
                    next_state, reward, done = env.step(action)
                    agent.memory.push(state, action, reward, next_state, done)
                    agent.learn()
                    state = next_state
                
                if stop_training: break

                if env.score > high_score:
                    high_score = env.score
                    save_high_score(high_score)
                
                # ⭐️ 이 부분 추가: 50 에피소드마다 프린트 출력 (속도 저하 사실상 0%)
                if ep % 50 == 0:
                    current_eps = config.EPS_END + (config.EPS_START - config.EPS_END) * \
                                  np.exp(-1. * agent.steps / config.EPS_DECAY)
                    print(f"🚀 Ep: {ep} | Score: {env.score} | Steps: {agent.steps} | Eps: {current_eps:.3f}")
                    agent.save("model.pth")

        elif mode == "TRAIN_FAST":
            print("Fast Training Started... Press 'ESC' on Pygame window to stop.")
            agent.load("model.pth")
            
            plt.ion()
            fig, ax1 = plt.subplots(figsize=(8, 5))
            ax2 = ax1.twinx()
            ep_history, score_history, eps_history = [], [], []
            
            stop_training = False
            for ep in range(5000):
                for event in pygame.event.get():
                    if event.type == pygame.QUIT:
                        running = False
                        stop_training = True
                    if event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE:
                        stop_training = True
                
                if stop_training: break

                state = env.reset()
                done = False
                while not done:
                    pygame.event.pump()
                    keys = pygame.key.get_pressed()
                    if keys[pygame.K_ESCAPE]:
                        stop_training = True
                        break

                    action = agent.act(state)
                    next_state, reward, done = env.step(action)
                    agent.memory.push(state, action, reward, next_state, done)
                    agent.learn()
                    state = next_state
                
                if stop_training: break

                ep_history.append(ep)
                score_history.append(env.score)
                current_eps = config.EPS_END + (config.EPS_START - config.EPS_END) * \
                              np.exp(-1. * agent.steps / config.EPS_DECAY)
                eps_history.append(current_eps)

                if env.score > high_score:
                    high_score = env.score
                    save_high_score(high_score)
                
                if ep % 10 == 0:
                    print(f"Ep: {ep} | Score: {env.score} | Steps: {agent.steps} | Eps: {current_eps:.2f}")
                    ax1.clear()
                    ax2.clear()
                    ax1.set_xlabel('Episode')
                    ax1.set_ylabel('Score', color='tab:blue')
                    ax1.plot(ep_history, score_history, color='tab:blue', alpha=0.6, label='Score')
                    ax2.set_ylabel('Epsilon', color='tab:red')
                    ax2.plot(ep_history, eps_history, color='tab:red', linestyle='--', label='Epsilon')
                    plt.title('Fast Training (Press ESC on Pygame Window to Menu)')
                    plt.draw()
                    plt.pause(0.01)

                if ep % 50 == 0: agent.save("model.pth")
            
            print("Stopping Fast Training... Saving model.")
            agent.save("model.pth")
            plt.close(fig)
            plt.ioff()
            mode = "MENU"

        else: # PLAY, TRAIN_VISUAL, AI_PLAY
            state = env.reset()
            done = False
            agent.load("model.pth")

            while not done:
                for event in pygame.event.get():
                    if event.type == pygame.QUIT: 
                        done = True
                        running = False
                    if event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE:
                        done = True
                
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