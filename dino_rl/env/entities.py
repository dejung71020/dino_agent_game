# dino_rl/env/entities.py
import pygame
import random
import math
import config

class Obstacle:
    def __init__(self, x, type_override=None, y_val=None):
        self.x = x
        self.type = type_override if type_override else random.choice(["cactus", "ptero"])
        
        if self.type == "cactus":
            self.y = 0
            self.width = random.choice([30, 60, 90])
            self.height = 50
        else: # 익룡 고도 세분화
            # 0: 하단 (점프 필수), 45: 중단 (숙기 필수), 100: 상단 (방관 가능/점프 시 위험)
            self.y = random.choice([0, 45, 100]) 
            self.width = 45
            self.height = 35

    def get_rect(self):
        # 중단 익룡(y=45)의 경우, 바닥에서 350-(45+35)=270 위치에 생성됨.
        # 서 있는 공룡(상단 300)은 부딪히고, 숙인 공룡(상단 325)은 통과함.
        return pygame.Rect(self.x, 350 - self.y - self.height, self.width, self.height)

class Coin:
    def __init__(self, x, y=None, is_king=False):
        self.x = x
        self.y = y if y is not None else random.choice([50, 120, 180])
        self.is_king = is_king
        self.value = config.REWARD_KING_COIN if is_king else config.REWARD_COIN
        self.radius = 12 if is_king else 8

    def update(self, speed, dino_x, dino_y):
        # 자석 효과: 공룡이 근처(150px)에 있으면 빨려 들어옴
        dist = math.hypot(self.x - dino_x, self.y - dino_y)
        if dist < 150:
            self.x += (dino_x - self.x) * 0.15
            self.y += (dino_y - self.y) * 0.15
        else:
            self.x -= speed

    def get_pos(self):
        return (int(self.x), 350 - int(self.y))