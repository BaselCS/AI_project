import pygame
import neat
import random
from assets import Assets
from const import *
from gamestate import GameState

class Dinosaur:
    
    
    def __init__(self, genome=None, config=None, genome_id=1):
        self.genome_id = genome_id
        self.image = Assets.RUNNING[0]
        self.dino_run = True
        self.dino_jump = False
        self.dino_crouch = False
        self.jump_vel = JUMP_VELOCITY
        self.crouching_height = COUCHING_HEIGHT
        self.standing_height = STAND_HEIGHT
        self.rect = pygame.Rect(DINO_X_POS, DINO_Y_POS, self.image.get_width(), self.image.get_height())
        self.color = (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))
        self.step_index = 0
        self.genome = genome
        self.net = neat.nn.FeedForwardNetwork.create(genome, config) if genome and config else None
        self.fitness = 0
        self.score=0
        
    def update(self,score):
        self.score=score
        if self.dino_run:
            self.run()
        if self.dino_jump:
            self.jump()
        if self.dino_crouch:
            self.crouch()
        if self.step_index >= 10:
            self.step_index = 0
    
    def jump(self):
        self.image = Assets.JUMPING
        
        # Adjust jump based on game speed (higher speed = faster jumps)
        self.jump_speed_factor = 1.0 + (GameState.game_speed * 0.02)  # Fine-tune this
        
        if self.dino_jump:
            self.rect.y -= (self.jump_vel * 4) * self.jump_speed_factor
            self.jump_vel -= 0.6 * self.jump_speed_factor
        
        if self.rect.y >= DINO_Y_POS:
            self.dino_jump = False
            self.dino_run = True
            self.jump_vel = JUMP_VELOCITY  # Reset jump velocity
            self.rect.y = DINO_Y_POS  # Reset position to ground level
            
    def run(self):
        self.image = Assets.RUNNING[self.step_index // 5]
        self.rect = pygame.Rect(DINO_X_POS, DINO_Y_POS, self.image.get_width(), self.image.get_height())
        self.step_index += 1
        
            
    def crouch(self):
        self.rect.height = self.crouching_height  # Shrink hitbox    
        self.image = Assets.Crouch[self.step_index % 2]
        self.rect.x = DINO_X_POS
        self.rect.y = DINO_Y_POS + (self.standing_height - self.crouching_height + 20)

        self.step_index += 1
        if self.step_index >= 10:
            self.step_index = 0         
        
    def draw(self, SCREEN,obstacles):
        # Draw the dinosaur sprite (standing or crouching)
        if self.dino_crouch:
            SCREEN.blit(Assets.Crouch[self.step_index // 5], (self.rect.x, self.rect.y + (self.standing_height - self.crouching_height - 50)))
        else:
            SCREEN.blit(Assets.RUNNING[self.step_index // 5], (self.rect.x, self.rect.y))
        
        # Debug: Draw hitbox (red for crouching, green for standing)
        hitbox_color = (255, 0, 0) if self.dino_crouch else (0, 255, 0)
        pygame.draw.rect(SCREEN, hitbox_color, self.rect, 2)
        
        
        # Debug: Draw AI "vision" lines to obstacles
        if obstacles and hasattr(self, 'color'):  # Only if obstacles exist and color is defined
                first = GameState.obstacles[0]
                second = GameState.obstacles[1] if len(GameState.obstacles) > 1 else first  # تجنب الخطأ

                # Draw line from dino's head to obstacle
                if self.dino_crouch:
                    # Adjust line origin when crouching (lower head position)
                    line_start = (self.rect.x + 54, self.rect.y + self.crouching_height - 5)
                else:
                    line_start = (self.rect.x + 54, self.rect.y + 12)
                
                pygame.draw.line(
                    SCREEN, 
                    self.color, 
                    line_start,
                    first.rect.center, 
                    2
                )
                
                pygame.draw.line(
                    SCREEN, 
                    self.color, 
                    line_start,
                    second.rect.center, 
                    2
                )
                
                
    def to_dict(self):
        return {
            'genome_id': self.genome_id,
            'color': self.color,
            'fitness': self.fitness,
            'jump_velocity': self.jump_vel,
            'position': (self.rect.x, self.rect.y)
        }
