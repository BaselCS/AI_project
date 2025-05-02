import pygame
import os
import random
import math
import sys
import neat

# Initialize pygame and load assets before anything else
pygame.init()

# Constants
SCREEN_WIDTH, SCREEN_HEIGHT = 1100, 600
FPS = 30
DINO_X_POS, DINO_Y_POS = 80, 310
JUMP_VELOCITY = 8.5
BACKGROUND_Y = 380
INITIAL_GAME_SPEED = 20
NUMBER_OF_GENERATIONS = 500

screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))

class Assets:
    
    @staticmethod
    def load():
        Assets.RUNNING = {
            0: pygame.image.load(os.path.join("Assets/Dino", "DinoRun1.png")),
            1: pygame.image.load(os.path.join("Assets/Dino", "DinoRun2.png"))
        }
        Assets.JUMPING = pygame.image.load(os.path.join("Assets/Dino", "DinoJump.png"))
        Assets.SMALL_CACTUS = {
            0: pygame.image.load(os.path.join("Assets/Cactus", "SmallCactus1.png")),
            1: pygame.image.load(os.path.join("Assets/Cactus", "SmallCactus2.png")),
            2: pygame.image.load(os.path.join("Assets/Cactus", "SmallCactus3.png"))
        }
        Assets.LARGE_CACTUS = {
            0: pygame.image.load(os.path.join("Assets/Cactus", "LargeCactus1.png")),
            1: pygame.image.load(os.path.join("Assets/Cactus", "LargeCactus2.png")),
            2: pygame.image.load(os.path.join("Assets/Cactus", "LargeCactus3.png"))
        }
        Assets.BACKGROUND = pygame.image.load(os.path.join("Assets/Other", "Track.png"))
        Assets.FONT = pygame.font.Font('freesansbold.ttf', 20)

# Load assets immediately when module is imported
Assets.load()

class Dinosaur:
    def __init__(self, image=Assets.RUNNING[0]):
        
        self.image = image
        self.dino_run = True
        self.dino_jump = False
        self.jump_vel = JUMP_VELOCITY
        self.rect = pygame.Rect(DINO_X_POS, DINO_Y_POS, image.get_width(), image.get_height())
        self.color = (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))
        self.step_index = 0
    
    def update(self):
        if self.dino_run:
            self.run()
        if self.dino_jump:
            self.jump()
        if self.step_index >= 10:
            self.step_index = 0
    
    def jump(self):
        
        self.image = Assets.JUMPING
        
        if self.dino_jump:
            self.rect.y -= self.jump_vel * 4
            self.jump_vel -= 0.8
        
        if self.jump_vel <= -JUMP_VELOCITY:
            self.dino_jump = False
            self.dino_run = True
            self.jump_vel = JUMP_VELOCITY
            
    def run(self):
        self.image = Assets.RUNNING[self.step_index // 5]
        self.rect.x = DINO_X_POS
        self.rect.y = DINO_Y_POS
        self.step_index += 1
    
    def draw(self, screen):
        
        screen.blit(self.image, (self.rect.x, self.rect.y))
        pygame.draw.rect(screen, self.color, self.rect, 2)
        
        for obstacle in GameState.obstacles:
            pygame.draw.line(screen, self.color, 
                           (self.rect.x + 54, self.rect.y + 12),
                           obstacle.rect.center, 2)

class Obstacle:
    def __init__(self, image, number_of_cacti):
        
        self.image = image
        self.type = number_of_cacti
        self.rect = self.image[self.type].get_rect()
        self.rect.x = SCREEN_WIDTH

    def update(self):
        self.rect.x -= GameState.game_speed
        return self.rect.x < -self.rect.width  # Returns True if off-screen
    
    def draw(self, screen):
        
        screen.blit(self.image[self.type], self.rect)

class SmallCactus(Obstacle):
    def __init__(self, image, number_of_cacti):
        super().__init__(image, number_of_cacti)
        self.rect.y = 325

class LargeCactus(Obstacle):
    def __init__(self, image, number_of_cacti):
        super().__init__(image, number_of_cacti)
        self.rect.y = 300

class GameState:
    """Centralized game state management"""
    obstacles = []
    dinosaurs = []
    gen_pool = []
    nets = []
    points = 0
    game_speed = INITIAL_GAME_SPEED
    x_pos_bg = 0
    y_pos_bg = BACKGROUND_Y
    spawn_cooldown = 2000  # ms
    last_spawn_time = 0
    population = None

    @staticmethod
    def reset():
        GameState.obstacles = []
        GameState.dinosaurs = []
        GameState.gen_pool = []
        GameState.nets = []
        GameState.points = 0
        GameState.game_speed = INITIAL_GAME_SPEED
        GameState.x_pos_bg = 0
        GameState.y_pos_bg = BACKGROUND_Y

def distance(pos_a, pos_b):
    dx = pos_a[0] - pos_b[0]
    dy = pos_a[1] - pos_b[1]
    return math.sqrt(dx**2 + dy**2)

def remove_dinosaur(index):
    GameState.dinosaurs.pop(index)
    GameState.gen_pool.pop(index)
    GameState.nets.pop(index)

def score(screen):
    GameState.points += 1
    if GameState.points % 100 == 0:
        GameState.game_speed += 1
    text = Assets.FONT.render(f"Points: {GameState.points}", True, (0, 0, 0))
    screen.blit(text, (950, 50))

def statistics(screen):
    text_1 = Assets.FONT.render(f'Dinosaurs Alive: {len(GameState.dinosaurs)}', True, (0, 0, 0))
    text_2 = Assets.FONT.render(f'Generation: {GameState.population.generation+1}', True, (0, 0, 0))
    text_3 = Assets.FONT.render(f'Game Speed: {GameState.game_speed}', True, (0, 0, 0))

    screen.blit(text_1, (50, 450))
    screen.blit(text_2, (50, 480))
    screen.blit(text_3, (50, 510))

def draw_background(screen):
    image_width = Assets.BACKGROUND.get_width()
    screen.blit(Assets.BACKGROUND, (GameState.x_pos_bg, GameState.y_pos_bg))
    screen.blit(Assets.BACKGROUND, (image_width + GameState.x_pos_bg, GameState.y_pos_bg))
    
    if GameState.x_pos_bg <= -image_width:
        GameState.x_pos_bg = 0
    GameState.x_pos_bg -= GameState.game_speed

def spawn_obstacle():
    
    current_time = pygame.time.get_ticks()
    if len(GameState.obstacles) < 3 and current_time - GameState.last_spawn_time+random.randint(100,1500)  > GameState.spawn_cooldown:
        if random.randint(0, 1) == 0:
            GameState.obstacles.append(SmallCactus(Assets.SMALL_CACTUS, random.randint(0, 2)))
        else:
            GameState.obstacles.append(LargeCactus(Assets.LARGE_CACTUS, random.randint(0, 2)))
        GameState.last_spawn_time = current_time

def eval_genomes(genomes, config):
    GameState.reset()
    clock = pygame.time.Clock()
    
    
    # Initialize NEAT population
    for genome_id, genome in genomes:
        GameState.dinosaurs.append(Dinosaur())
        GameState.gen_pool.append(genome)
        GameState.nets.append(neat.nn.FeedForwardNetwork.create(genome, config))
        genome.fitness = 0

    while True:
        if not GameState.dinosaurs:
            break

        screen.fill((255, 255, 255))
        
        # Game logic
        for dinosaur in GameState.dinosaurs:
            dinosaur.update()
            dinosaur.draw(screen)
        
        if not GameState.dinosaurs:
            break
        
        # Obstacle management
        spawn_obstacle()
        GameState.obstacles = [obstacle for obstacle in GameState.obstacles if not obstacle.update()]
        
        for obstacle in GameState.obstacles:
            obstacle.draw(screen)
            for i, dinosaur in enumerate(GameState.dinosaurs):
                if dinosaur.rect.colliderect(obstacle.rect):
                    GameState.gen_pool[i].fitness -= 1
                    remove_dinosaur(i)
                    
        
        # AI decision making
        for i, dinosaur in enumerate(GameState.dinosaurs):
            if GameState.obstacles:
                output = GameState.nets[i].activate((
                    dinosaur.rect.y,
                    distance((dinosaur.rect.x, dinosaur.rect.y),
                             GameState.obstacles[0].rect.midtop)
                ))
                if output[0] > 0.5 and dinosaur.rect.y == DINO_Y_POS:
                    dinosaur.dino_jump = True
                    dinosaur.dino_run = False
        
        # Drawing
        statistics(screen)
        score(screen)
        draw_background(screen)
        
        pygame.display.update()
        clock.tick(FPS)

def run(config_path):
    screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
    config = neat.config.Config(
        neat.DefaultGenome,
        neat.DefaultReproduction,
        neat.DefaultSpeciesSet,
        neat.DefaultStagnation,
        config_path
    )
    
    GameState.population = neat.Population(config)
    GameState.population.run(eval_genomes, NUMBER_OF_GENERATIONS)

if __name__ == '__main__':
    # Assets are already loaded at module level
    local_dir = os.path.dirname(__file__)
    config_path = os.path.join(local_dir, 'config.txt')
    run(config_path)