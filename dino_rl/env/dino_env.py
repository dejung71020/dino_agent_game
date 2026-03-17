# dino_rl/env/dino_env.py
import numpy as np
import pygame
import random
import config
import time
from env.entities import Obstacle, Coin

class DinoEnv:
    def __init__(self):
        self.reset()

    def reset(self):
        self.dino_y = 0
        self.dino_vel = 0
        self.is_jumping = False
        self.is_ducking = False
        self.speed = config.START_SPEED
        self.obstacles = [Obstacle(800)]
        self.coins = []
        self.score = 0
        self.start_time = time.time()  # 게임 시작 시간 기록
        return self.get_state()

    def get_elapsed_time(self):
        return int(time.time() - self.start_time)

    def step(self, action):
        reward = config.REWARD_SURVIVE
        done = False
        
        if action == 1 and not self.is_jumping:
            self.dino_vel = config.JUMP_VELOCITY
            self.is_jumping = True
        self.is_ducking = (action == 2)
        
        grav = config.DUCK_GRAVITY if self.is_ducking else config.GRAVITY
        self.dino_vel += grav
        self.dino_y -= self.dino_vel
        if self.dino_y <= 0:
            self.dino_y = 0
            self.dino_vel = 0
            self.is_jumping = False

        h = 25 if self.is_ducking else 50
        w = 60 if self.is_ducking else 40
        dino_rect = pygame.Rect(50, 350 - self.dino_y - h, w, h)

        for obs in self.obstacles:
            obs.x -= self.speed
            if dino_rect.colliderect(obs.get_rect()):
                done = True
                reward = config.REWARD_DEATH
            elif 0 < obs.x - 50 < 30 and not done:
                reward += config.REWARD_NEAR_MISS / 60

        for c in self.coins[:]:
            c.update(self.speed, 50, self.dino_y)
            if dino_rect.collidepoint(c.get_pos()):
                reward += c.value
                self.coins.remove(c)
            elif c.x < -20:
                self.coins.remove(c)

        if self.obstacles[-1].x < 600:
            self.spawn_manager()
            reward += config.REWARD_PASS

        if self.obstacles[0].x < -100: self.obstacles.pop(0)
        
        self.speed = min(config.MAX_SPEED, self.speed + config.SPEED_INCREMENT)
        self.score += 1
        return self.get_state(), reward, done

    def spawn_manager(self):
        # 장애물 간 최소 안전거리 확보 (최소 400px 이상)
        new_x = self.obstacles[-1].x + random.randint(450, 700) 
        
        rand = random.random()
        # 익룡 등장 확률을 40%로 상향하여 숙이기 기회 제공
        if rand < 0.4:
            self.obstacles.append(Obstacle(new_x, "ptero"))
        # 10% 확률로 코인 함정 패턴 생성
        elif rand < 0.5:
            for i in range(3): self.coins.append(Coin(new_x + i*40, 150))
            self.obstacles.append(Obstacle(new_x + 150, "cactus"))
        else:
            self.obstacles.append(Obstacle(new_x, "cactus"))
            
        # 코인 배치 (장애물과 겹치지 않게 보정)
        if random.random() < 0.2:
            self.coins.append(Coin(new_x - 150))

    def get_state(self):
        # 공룡의 앞부분(대략 x=40)을 기준으로, 이미 지나간 장애물은 시야에서 제외
        active_obs = [o for o in self.obstacles if o.x + o.width > 40]
        
        obs = active_obs[0] if len(active_obs) > 0 else Obstacle(2000)
        next_obs = active_obs[1] if len(active_obs) > 1 else Obstacle(2000)
        
        coin = self.coins[0] if self.coins else Coin(2000)
        
        # 💡 핵심 추가: Time-To-Collision (현재 속도 기반 충돌 예상 시간)
        # 거리가 멀어도 속도가 빠르면 위험하다는 것을 AI가 인지하게 됩니다.
        ttc1 = max(0, obs.x - 50) / self.speed / 100.0
        
        state = [
            self.dino_y / 200, self.dino_vel / 20, self.speed / config.MAX_SPEED,
            
            # 첫 번째 장애물 정보
            (obs.x - 50) / 1000, obs.y / 200, obs.width / 100, (1 if obs.type == "ptero" else 0),
            
            # 두 번째 장애물 정보
            (next_obs.x - 50) / 1000, next_obs.y / 200, next_obs.width / 100, (1 if next_obs.type == "ptero" else 0),
            
            # 코인 및 공룡 상태 정보
            (coin.x - 50) / 1000, (coin.y - self.dino_y) / 200,
            int(self.is_ducking), int(self.is_jumping), 
            
            ttc1 # 💡 기존 0(패딩)이었던 자리에 TTC 정보 투입! (16차원 유지)
        ]
        return np.array(state, dtype=np.float32)