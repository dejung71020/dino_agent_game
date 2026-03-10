# dino_rl/render/renderer.py
import pygame
import torch
import config
import os

class Renderer:
    def __init__(self):
        pygame.init()
        self.screen = pygame.display.set_mode((config.SCREEN_WIDTH, config.SCREEN_HEIGHT))
        self.clock = pygame.time.Clock()
        self.font_main = pygame.font.SysFont("Arial", 22, bold=True)
        self.font_big = pygame.font.SysFont("Arial", 40, bold=True)
        self.trails = []
        
        # 이미지 자산 로드 (없으면 기본 도형으로 대체됨)
        self.assets = {}
        self.load_assets()

    def load_assets(self):
        # 만약 assets 폴더에 아래 이름의 파일이 있으면 불러옵니다.
        asset_names = ['dino', 'cactus', 'ptero', 'coin', 'cloud', 'ground']
        for name in asset_names:
            path = f"assets/{name}.png"
            if os.path.exists(path):
                self.assets[name] = pygame.image.load(path).convert_alpha()

    def draw_menu(self):
        # 세련된 다크 테마 메뉴
        self.screen.fill((45, 52, 71))
        title_txt = self.font_big.render("DINO RL MASTERPIECE", True, (255, 255, 255))
        self.screen.blit(title_txt, (config.SCREEN_WIDTH//2 - title_txt.get_width()//2, 50))
        
        menus = [
            "1. MANUAL PLAY (CHALLENGE)",
            "2. VISUAL TRAINING (WATCH AI)",
            "3. FAST TRAINING (SPEED UP)",
            "4. AI AUTO PLAY (BEST MODEL)"
        ]
        for i, m in enumerate(menus):
            color = (200, 200, 200)
            rect = pygame.Rect(350, 150 + i*50, 300, 40)
            pygame.draw.rect(self.screen, (60, 70, 90), rect, border_radius=10)
            txt = self.font_main.render(m, True, color)
            self.screen.blit(txt, (rect.x + 20, rect.y + 8))
        
        pygame.display.flip()

    def draw_game(self, env, agent=None, mode_info="", high_score=0):
        # 배경 그리기 (하늘색 그라데이션 효과)
        self.screen.fill((235, 245, 255))
        
        # 땅 그리기
        pygame.draw.rect(self.screen, (139, 69, 19), (0, 350, config.SCREEN_WIDTH, 50)) # 흙
        pygame.draw.line(self.screen, (34, 139, 34), (0, 350), (config.SCREEN_WIDTH, 350), 4) # 잔디 선
        
        # 잔상 (귀여운 효과)
        self.trails.append((50, 350 - env.dino_y, env.is_ducking))
        if len(self.trails) > 12: self.trails.pop(0)
        for i, (tx, ty, td) in enumerate(self.trails):
            alpha = int(180 * (i/12))
            s = pygame.Surface((40 if not td else 60, 50 if not td else 25), pygame.SRCALPHA)
            pygame.draw.rect(s, (100, 200, 255, alpha), s.get_rect(), border_radius=5)
            self.screen.blit(s, (tx, ty - (25 if td else 50)))

        # 공룡 그리기 (이미지가 없으면 눈이 달린 귀여운 사각형으로 그림)
        d_h = 25 if env.is_ducking else 50
        d_w = 60 if env.is_ducking else 40
        d_rect = pygame.Rect(50, 350 - env.dino_y - d_h, d_w, d_h)
        if 'dino' in self.assets:
            img = pygame.transform.scale(self.assets['dino'], (d_w, d_h))
            self.screen.blit(img, d_rect)
        else:
            pygame.draw.rect(self.screen, (50, 180, 100), d_rect, border_radius=8) # 몸통
            pygame.draw.circle(self.screen, (255, 255, 255), (d_rect.right - 10, d_rect.top + 10), 4) # 눈

        # 장애물 그리기
        for obs in env.obstacles:
            o_rect = obs.get_rect()
            if obs.type == 'cactus':
                pygame.draw.rect(self.screen, (34, 139, 34), o_rect, border_radius=3)
            else: # ptero
                pygame.draw.ellipse(self.screen, (200, 80, 80), o_rect)
        
        # 코인 그리기
        for c in env.coins:
            color = (255, 215, 0) if not c.is_king else (255, 100, 0)
            pygame.draw.circle(self.screen, color, c.get_pos(), c.radius)
            pygame.draw.circle(self.screen, (255, 255, 255), c.get_pos(), c.radius-3, 1)

        # 상단 UI (점수, 최고기록, 시간)
        elapsed = env.get_elapsed_time()
        score_txt = self.font_main.render(f"SCORE: {env.score:05}", True, (50, 50, 50))
        high_txt = self.font_main.render(f"HI-SCORE: {high_score:05}", True, (150, 50, 50))
        time_txt = self.font_main.render(f"TIME: {elapsed}s", True, (50, 50, 150))
        
        self.screen.blit(score_txt, (20, 20))
        self.screen.blit(high_txt, (20, 50))
        self.screen.blit(time_txt, (20, 80))
        
        # 모드 정보 표기
        mode_txt = self.font_main.render(f"MODE: {mode_info}", True, (100, 100, 100))
        self.screen.blit(mode_txt, (config.SCREEN_WIDTH - 200, 20))

        # AI HUD (기존 기능 유지)
        if agent and mode_info != "TRAIN_FAST":
            self.draw_ai_hud(env, agent)

        pygame.display.flip()
        self.clock.tick(config.FPS)

    def draw_ai_hud(self, env, agent):
        # AI 생각 시각화
        state_t = torch.FloatTensor(env.get_state()).unsqueeze(0).to(agent.device)
        with torch.no_grad(): 
            q_vals = agent.online_net(state_t).cpu().numpy()[0]
        
        actions = ["RUN", "JUMP", "DUCK"]
        for i, q in enumerate(q_vals):
            width = int(abs(q) * 4)
            bar_rect = pygame.Rect(800, 300 + i*25, width, 20)
            pygame.draw.rect(self.screen, (100, 100, 250), bar_rect, border_radius=5)
            self.screen.blit(self.font_main.render(actions[i], True, (0,0,0)), (730, 300 + i*25))