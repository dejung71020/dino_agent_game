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
                    elif event.key == pygame.K_2: mode = "TRAIN_CHART"
                    elif event.key == pygame.K_3: mode = "TRAIN_TEXT"
                    elif event.key == pygame.K_4: mode = "TRAIN_MAX"
                    elif event.key == pygame.K_5: mode = "AI_PLAY"
                    elif event.key == pygame.K_ESCAPE: running = False

        elif mode == "TRAIN_CHART":
            print("📊 차트 학습 모드 시작. Pygame 창에서 ESC를 누르면 메뉴로 돌아갑니다.")
            agent.load("model.pth")
            plt.ion()
            fig, ax1 = plt.subplots(figsize=(8, 5))
            ax2 = ax1.twinx()
            ep_history, score_history, eps_history = [], [], []
            
            stop_training = False
            ep = 1
            while not stop_training and ep <= 50000:
                state = env.reset()
                done = False
                while not done:
                    for event in pygame.event.get():
                        if event.type == pygame.QUIT:
                            running = False; stop_training = True; break
                        if event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE:
                            stop_training = True; break
                    if stop_training: break

                    action = agent.act(state)
                    next_state, reward, done = env.step(action)
                    agent.memory.push(state, action, reward, next_state, done)
                    agent.learn()
                    state = next_state
                    
                    if env.score > high_score:
                        high_score = env.score
                        save_high_score(high_score)
                    
                    renderer.draw_game(env, agent, "TRAIN_CHART", high_score, fps=0)
                
                if stop_training: break

                ep_history.append(ep)
                score_history.append(env.score)
                current_eps = config.EPS_END + (config.EPS_START - config.EPS_END) * np.exp(-1. * agent.steps / config.EPS_DECAY)
                eps_history.append(current_eps)

                if ep % 10 == 0:
                    try:
                        ax1.clear(); ax2.clear()
                        ax1.set_xlabel('Episode'); ax1.set_ylabel('Score', color='tab:blue')
                        ax1.plot(ep_history, score_history, color='tab:blue', alpha=0.6)
                        ax2.set_ylabel('Epsilon', color='tab:red')
                        ax2.plot(ep_history, eps_history, color='tab:red', linestyle='--')
                        plt.title('AI Training Progress')
                        
                        fig.canvas.draw()
                        fig.canvas.flush_events()
                    except Exception as e:
                        pass

                if ep % 50 == 0: agent.save("model.pth")
                ep += 1
            
            plt.close(fig); plt.ioff()
            agent.save("model.pth")
            mode = "MENU"

        elif mode == "TRAIN_MAX":
            print("🚀 최대 효율 학습 모드 가동 중... (ESC: 메뉴로)")
            agent.load("model.pth")
            stop_training = False
            
            ep = 1
            while not stop_training and ep <= 100000:
                state = env.reset()
                done = False
                while not done:
                    # 100스텝마다 한 번씩만 키 입력 체크 (연산 효율 극대화)
                    if agent.steps % 100 == 0:
                        pygame.event.pump()
                        if pygame.key.get_pressed()[pygame.K_ESCAPE]:
                            stop_training = True; break

                    action = agent.act(state)
                    next_state, reward, done = env.step(action)
                    agent.memory.push(state, action, reward, next_state, done)
                    
                    # 💡 제안드린 config.LEARN_EVERY 주기에 맞춰 집중 학습
                    if agent.steps % config.LEARN_EVERY == 0:
                        for _ in range(config.LEARN_STEPS):
                            agent.learn()
                            
                    state = next_state
                
                if stop_training: break

                if env.score > high_score:
                    high_score = env.score
                    save_high_score(high_score)
                
                # 출력 빈도를 조절하여 콘솔 I/O 병목 방지
                if ep % 50 == 0:
                    current_eps = config.EPS_END + (config.EPS_START - config.EPS_END) * \
                                  np.exp(-1. * agent.steps / config.EPS_DECAY)
                    print(f"🚀 [MAX] Ep: {ep} | Score: {env.score} | Total Steps: {agent.steps} | Eps: {current_eps:.3f}")
                    agent.save("model.pth")
                ep += 1
            
            agent.save("model.pth")
            mode = "MENU"

        else: # PLAY, TRAIN_TEXT, AI_PLAY
            state = env.reset()
            agent.load("model.pth")
            ep_count = 0
            game_over_paused = False  # 💡 게임 오버 상태를 체크하는 변수 추가
            
            while running and mode in ["PLAY", "TRAIN_TEXT", "AI_PLAY"]:
                for event in pygame.event.get():
                    if event.type == pygame.QUIT: 
                        running = False
                    if event.type == pygame.KEYDOWN:
                        if event.key == pygame.K_ESCAPE:
                            game_over_paused = False
                            mode = "MENU" # ESC를 누르면 메뉴로 탈출
                        elif game_over_paused and event.key in [pygame.K_SPACE, pygame.K_RETURN]:
                            # 💡 게임 오버 상태에서 스페이스바나 엔터를 누르면 재시작
                            game_over_paused = False
                            state = env.reset()

                if mode == "MENU" or not running:
                    break

                # 💡 게임 오버 상태면 게임 로직(step)을 돌리지 않고 화면만 유지
                if game_over_paused:
                    renderer.draw_game_over(env.score, high_score)
                    renderer.clock.tick(15) # UI 대기 시엔 리소스를 덜 먹도록 프레임을 낮춤
                    continue

                if mode == "PLAY":
                    keys = pygame.key.get_pressed()
                    action = 1 if keys[pygame.K_SPACE] or keys[pygame.K_UP] else (2 if keys[pygame.K_DOWN] else 0)
                elif mode == "TRAIN_TEXT":
                    action = agent.act(state)
                else: # AI_PLAY
                    action = agent.act(state, train=False)

                next_state, reward, done = env.step(action)
                
                if mode == "TRAIN_TEXT":
                    agent.memory.push(state, action, reward, next_state, done)
                    agent.learn()
                
                if env.score > high_score:
                    high_score = env.score
                    save_high_score(high_score)
                
                renderer.draw_game(env, agent, mode, high_score, fps=config.FPS)
                state = next_state
                
                if done:
                    ep_count += 1
                    if mode == "TRAIN_TEXT":
                        current_eps = config.EPS_END + (config.EPS_START - config.EPS_END) * np.exp(-1. * agent.steps / config.EPS_DECAY)
                        print(f"📄 [TEXT] Ep: {ep_count} | Score: {env.score} | Eps: {current_eps:.3f}")
                        if ep_count % 20 == 0: agent.save("model.pth")
                        state = env.reset() # 텍스트 모드는 즉시 재시작
                    else:
                        # 💡 직접 플레이나 AI 플레이 모드에서는 게임 오버 화면 띄우기
                        game_over_paused = True
                        renderer.draw_game_over(env.score, high_score)

            if mode == "MENU" and mode == "TRAIN_TEXT": 
                agent.save("model.pth")

    pygame.quit()

if __name__ == "__main__":
    main()