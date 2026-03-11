# dino_rl/render/renderer.py
import pygame
import torch
import config
import os
import numpy as np

# 모던 구글 스타일 색상 테마
WHITE = (255, 255, 255)
BG_COLOR = (248, 249, 250)
CARD_BG = (255, 255, 255)
TEXT_MAIN = (32, 33, 36)
TEXT_SUB = (95, 99, 104)
ACCENT_BLUE = (26, 115, 232)
GROUND_COLOR = (218, 220, 224)

class Renderer:
    def __init__(self):
        pygame.init()
        self.screen = pygame.display.set_mode((config.SCREEN_WIDTH, config.SCREEN_HEIGHT))
        pygame.display.set_caption("Dino RL Masterpiece")
        self.clock = pygame.time.Clock()
        
        # 폰트 설정
        self.font_title = pygame.font.SysFont("Arial", 42, bold=True)
        self.font_main = pygame.font.SysFont("Arial", 22, bold=True)
        self.font_small = pygame.font.SysFont("Arial", 16, bold=True)
        
        self.assets = {}
        self.load_assets()

    def load_assets(self):
        # 절대 경로 설정
        base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        assets_dir = os.path.join(base_dir, "assets")

        asset_names = [
            'dino', 'dino_run1', 'dino_run2', 
            'dino_duck1', 'dino_duck2', 
            'cactus', 'ptero', 'coin'
        ]
        
        for name in asset_names:
            png_path = os.path.join(assets_dir, f"{name}.png")
            jpg_path = os.path.join(assets_dir, f"{name}.jpg")
            
            img_to_load = None
            if os.path.exists(png_path):
                img_to_load = pygame.image.load(png_path).convert_alpha()
            elif os.path.exists(jpg_path):
                img_to_load = pygame.image.load(jpg_path).convert_alpha()
                
            if img_to_load:
                # 💡 핵심 마법: 이미지 주변의 투명한 빈 공간(여백)을 자동으로 싹둑 잘라냅니다!
                bounding_rect = img_to_load.get_bounding_rect()
                # 빈 공간을 잘라낸 알맹이 이미지만 에셋으로 저장
                self.assets[name] = img_to_load.subsurface(bounding_rect)
            else:
                self.assets[name] = None

    def draw_menu(self):
        self.screen.fill(BG_COLOR)
        
        title_txt = self.font_title.render("Dino RL Masterpiece", True, ACCENT_BLUE)
        self.screen.blit(title_txt, (config.SCREEN_WIDTH//2 - title_txt.get_width()//2, 60))
        
        sub_txt = self.font_small.render("Select mode to start", True, TEXT_SUB)
        self.screen.blit(sub_txt, (config.SCREEN_WIDTH//2 - sub_txt.get_width()//2, 110))
        
        menus = [
            ("1", "play (Manual Play)"),
            ("2", "chart ai trainning (Train + Chart)"),
            ("3", "text ai trainning (Train + Text Log)"),
            ("4", "AI trainning (No Rendering, Fastest)"),
            ("5", "AI play (AI Auto Play)")
        ]
        
        for i, (key, text) in enumerate(menus):
            rect = pygame.Rect(config.SCREEN_WIDTH//2 - 200, 160 + i*45, 400, 40)
            pygame.draw.rect(self.screen, CARD_BG, rect, border_radius=8)
            pygame.draw.rect(self.screen, GROUND_COLOR, rect, 1, border_radius=8)
            
            key_bg = pygame.Rect(rect.x + 10, rect.y + 8, 24, 24)
            pygame.draw.rect(self.screen, ACCENT_BLUE, key_bg, border_radius=4)
            key_txt = self.font_small.render(key, True, WHITE)
            self.screen.blit(key_txt, (key_bg.x + 8, key_bg.y + 3))
            
            menu_txt = self.font_main.render(text, True, TEXT_MAIN)
            self.screen.blit(menu_txt, (rect.x + 50, rect.y + 8))
        
        pygame.display.flip()

    def draw_game(self, env, agent=None, mode_info="", high_score=0, fps=config.FPS):
        self.screen.fill(BG_COLOR)
        
        # 바닥 선
        pygame.draw.line(self.screen, GROUND_COLOR, (0, 350), (config.SCREEN_WIDTH, 350), 3)

        # 1. 공룡 렌더링 (충돌 박스 100% 꽉 채우기)
        d_h = 25 if env.is_ducking else 50
        d_w = 60 if env.is_ducking else 40
        d_rect = pygame.Rect(50, 350 - env.dino_y - d_h, d_w, d_h)
        
        is_frame_1 = (env.score // 10) % 2 == 0 
        
        sprite_name = 'dino'
        if env.is_ducking:
            sprite_name = 'dino_duck1' if is_frame_1 else 'dino_duck2'
        elif env.dino_y > 0 or env.is_jumping:
            sprite_name = 'dino'
        else:
            sprite_name = 'dino_run1' if is_frame_1 else 'dino_run2'

        img_loaded = self.assets.get(sprite_name)
        if img_loaded:
            # 💡 이미지를 실제 충돌 박스 크기에 무조건 맞춥니다.
            img = pygame.transform.scale(img_loaded, (d_w, d_h))
            self.screen.blit(img, d_rect)
        else:
            pygame.draw.rect(self.screen, TEXT_SUB, d_rect, border_radius=6)

        # 2. 장애물 렌더링
        for obs in env.obstacles:
            o_rect = obs.get_rect()
            obs_img = self.assets.get(obs.type)
            
            if obs_img:
                if obs.type == 'cactus':
                    # 💡 바닥 장애물(스테고) 넓이에 따라 여러 마리로 복제해서 그림 (충돌 박스와 100% 일치)
                    num_cacti = max(1, o_rect.width // 30)
                    single_w = o_rect.width // num_cacti
                    for i in range(num_cacti):
                        img = pygame.transform.scale(obs_img, (single_w, o_rect.height))
                        part_rect = pygame.Rect(o_rect.x + i * single_w, o_rect.y, single_w, o_rect.height)
                        self.screen.blit(img, part_rect)
                else:
                    # 익룡은 충돌 박스 크기에 딱 맞게 스케일링
                    img = pygame.transform.scale(obs_img, (o_rect.width, o_rect.height))
                    self.screen.blit(img, o_rect)
            else:
                if obs.type == 'cactus':
                    pygame.draw.rect(self.screen, (34, 139, 34), o_rect, border_radius=3)
                else:
                    pygame.draw.ellipse(self.screen, (200, 80, 80), o_rect)

        # 3. 코인 렌더링
        coin_img = self.assets.get('coin')
        for c in env.coins:
            c_rect = pygame.Rect(c.x - c.radius, 350 - c.y - c.radius, c.radius*2, c.radius*2)
            if coin_img:
                img = pygame.transform.scale(coin_img, (c_rect.width, c_rect.height))
                self.screen.blit(img, c_rect)
            else:
                color = (255, 215, 0) if not c.is_king else (255, 100, 0)
                pygame.draw.circle(self.screen, color, c.get_pos(), c.radius)

        # 4. 상단 UI
        score_txt = self.font_main.render(f"SCORE: {env.score:05}", True, TEXT_MAIN)
        high_txt = self.font_main.render(f"HI: {high_score:05}", True, TEXT_SUB)
        mode_txt = self.font_small.render(f"MODE: {mode_info}", True, ACCENT_BLUE)
        
        self.screen.blit(score_txt, (20, 20))
        self.screen.blit(high_txt, (20, 50))
        self.screen.blit(mode_txt, (config.SCREEN_WIDTH - 200, 20))

        # 5. AI HUD
        if agent and mode_info not in ["PLAY", "TRAIN_MAX"]:
            self.draw_ai_hud(env, agent)

        pygame.display.flip()
        
        if fps > 0:
            self.clock.tick(fps)

    def draw_ai_hud(self, env, agent):
        state_t = torch.FloatTensor(env.get_state()).unsqueeze(0).to(agent.device)
        with torch.no_grad(): 
            q_vals = agent.online_net(state_t).cpu().numpy()[0]
        
        actions = ["RUN", "JUMP", "DUCK"]
        hud_bg = pygame.Rect(config.SCREEN_WIDTH - 200, 50, 180, 90)
        pygame.draw.rect(self.screen, WHITE, hud_bg, border_radius=8)
        pygame.draw.rect(self.screen, GROUND_COLOR, hud_bg, 1, border_radius=8)

        for i, q in enumerate(q_vals):
            width = min(int(abs(q) * 4), 100)
            bar_rect = pygame.Rect(config.SCREEN_WIDTH - 130, 60 + i*25, width, 16)
            
            color = ACCENT_BLUE if q == max(q_vals) else GROUND_COLOR
            pygame.draw.rect(self.screen, color, bar_rect, border_radius=4)
            
            txt = self.font_small.render(actions[i], True, TEXT_MAIN)
            self.screen.blit(txt, (config.SCREEN_WIDTH - 190, 60 + i*25))

    def draw_game_over(self, score, high_score):
        overlay = pygame.Surface((config.SCREEN_WIDTH, config.SCREEN_HEIGHT))
        overlay.set_alpha(180)
        overlay.fill(WHITE)
        self.screen.blit(overlay, (0, 0))

        go_txt = self.font_title.render("GAME OVER", True, (217, 48, 37))
        self.screen.blit(go_txt, (config.SCREEN_WIDTH//2 - go_txt.get_width()//2, 120))

        score_txt = self.font_main.render(f"Score: {score}   |   High Score: {high_score}", True, TEXT_MAIN)
        self.screen.blit(score_txt, (config.SCREEN_WIDTH//2 - score_txt.get_width()//2, 180))

        restart_txt = self.font_main.render("Press SPACE or ENTER to Restart / ESC to Menu", True, TEXT_SUB)
        self.screen.blit(restart_txt, (config.SCREEN_WIDTH//2 - restart_txt.get_width()//2, 240))

        pygame.display.flip()