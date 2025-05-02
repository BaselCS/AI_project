import math
import random
import pygame
import os
import json
import neat
from neat.nn import FeedForwardNetwork

# Initialize pygame
pygame.init()

# Constants (should match your training constants)
SCREEN_WIDTH, SCREEN_HEIGHT = 1200, 900
FPS = 30
DINO_X_POS, DINO_Y_POS = 80, 310
JUMP_VELOCITY = 8.5
BACKGROUND_Y = 380
INITIAL_GAME_SPEED = 20
SAVE_DIR = "dino_saves"

SCREEN = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))

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

Assets.load()

class Dinosaur:
    def __init__(self, genome=None, config=None):
        self.image = Assets.RUNNING[0]
        self.dino_run = True
        self.dino_jump = False
        self.jump_vel = JUMP_VELOCITY
        self.rect = pygame.Rect(DINO_X_POS, DINO_Y_POS, self.image.get_width(), self.image.get_height())
        self.color = (0, 255, 0)  # Green for the best dino
        self.step_index = 0
        self.genome = genome
        self.net = FeedForwardNetwork.create(genome, config) if genome and config else None
        self.fitness = 0
    
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
    
    def draw(self, SCREEN):
        SCREEN.blit(self.image, (self.rect.x, self.rect.y))
        pygame.draw.rect(SCREEN, self.color, (self.rect.x, self.rect.y, self.rect.width, self.rect.height), 2)

class Obstacle:
    def __init__(self, image, number_of_cacti):
        self.image = image
        self.type = number_of_cacti
        self.rect = self.image[self.type].get_rect()
        self.rect.x = SCREEN_WIDTH

    def update(self, game_speed):
        self.rect.x -= game_speed
        return self.rect.x < -self.rect.width
    
    def draw(self, SCREEN):
        SCREEN.blit(self.image[self.type], self.rect)

class SmallCactus(Obstacle):
    def __init__(self, image, number_of_cacti):
        super().__init__(image, number_of_cacti)
        self.rect.y = 325

class LargeCactus(Obstacle):
    def __init__(self, image, number_of_cacti):
        super().__init__(image, number_of_cacti)
        self.rect.y = 300

def distance(pos_a, pos_b):
    dx = pos_a[0] - pos_b[0]
    dy = pos_a[1] - pos_b[1]
    return math.sqrt(dx**2 + dy**2)

def load_best_dino(config_path):
    # Load the config file
    config = neat.config.Config(
        neat.DefaultGenome,
        neat.DefaultReproduction,
        neat.DefaultSpeciesSet,
        neat.DefaultStagnation,
        config_path
    )
    
    # Find the best generation
    summary_path = os.path.join(SAVE_DIR, "generations_summary.json")
    if not os.path.exists(summary_path):
        raise FileNotFoundError("No generations summary found. Train the AI first.")
    
    with open(summary_path) as f:
        generations = json.load(f)
    
    if not generations:
        raise ValueError("No generations data available.")
    
    # Get the best generation
    best_gen = max(generations.items(), key=lambda x: x[1]['fitness'])
    gen_num, best_data = best_gen
    print(f"Loading best dinosaur from generation {gen_num} with fitness {best_data['fitness']}")
    
    # Load the genome
    gen_dir = os.path.join(SAVE_DIR, f"gen_{gen_num}")
    best_dino_path = os.path.join(gen_dir, "best_dino.json")
    
    with open(best_dino_path) as f:
        dino_data = json.load(f)
    
    # Recreate the genome
    genome = neat.DefaultGenome(dino_data['genome_id'])
    genome.fitness = dino_data['fitness']
    
    # Recreate nodes with all attributes
    for node_data in dino_data['nodes']:
        node_id = node_data['id']
        genome.nodes[node_id] = neat.genome.DefaultNodeGene(node_id)
        genome.nodes[node_id].bias = node_data['bias']
        genome.nodes[node_id].activation = node_data['activation']
        genome.nodes[node_id].aggregation = node_data['aggregation']
        genome.nodes[node_id].response = node_data['response']
    
    # Recreate connections
    for conn_data in dino_data['connections']:
        key = (conn_data['in'], conn_data['out'])
        genome.connections[key] = neat.genome.DefaultConnectionGene(key)
        genome.connections[key].weight = conn_data['weight']
        genome.connections[key].enabled = conn_data['enabled']
    
    return genome, config

def test_best_dino():
    # Load NEAT config
    local_dir = os.path.dirname(__file__)
    config_path = os.path.join(local_dir, 'config.txt')
    
    try:
        genome, config = load_best_dino(config_path)
    except Exception as e:
        print(f"Error loading best dinosaur: {e}")
        return
    
    # Create the best dinosaur
    best_dino = Dinosaur(genome, config)
    
    # Game state
    obstacles = []
    points = 0
    game_speed = INITIAL_GAME_SPEED
    x_pos_bg = 0
    y_pos_bg = BACKGROUND_Y
    spawn_cooldown = 2000  # ms
    last_spawn_time = 0
    clock = pygame.time.Clock()
    running = True
    
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_SPACE:
                    best_dino.dino_jump = True
                    best_dino.dino_run = False
        
        SCREEN.fill((255, 255, 255))
        
        # Spawn obstacles
        current_time = pygame.time.get_ticks()
        if len(obstacles) < 3 and current_time - last_spawn_time + random.randint(100, 1500) > spawn_cooldown:
            if random.randint(0, 1) == 0:
                obstacles.append(SmallCactus(Assets.SMALL_CACTUS, random.randint(0, 2)))
            else:
                obstacles.append(LargeCactus(Assets.LARGE_CACTUS, random.randint(0, 2)))
            last_spawn_time = current_time
        
        # Update obstacles
        obstacles = [obstacle for obstacle in obstacles if not obstacle.update(game_speed)]
        
        # AI decision making
        if best_dino.net and obstacles:
            output = best_dino.net.activate((
                best_dino.rect.y,
                distance((best_dino.rect.x, best_dino.rect.y),
                         obstacles[0].rect.midtop)
            ))
            if output[0] > 0.5 and best_dino.rect.y == DINO_Y_POS:
                best_dino.dino_jump = True
                best_dino.dino_run = False
        
        # Update dinosaur
        best_dino.update()
        
        # Check for collisions
        for obstacle in obstacles:
            if best_dino.rect.colliderect(obstacle.rect):
                print(f"Game Over! Score: {points}")
                running = False
        
        # Score
        points += 1
        if points % 100 == 0:
            game_speed += 1
        
        # Draw everything
        for obstacle in obstacles:
            obstacle.draw(SCREEN)
        
        best_dino.draw(SCREEN)
        
        # Draw background
        image_width = Assets.BACKGROUND.get_width()
        SCREEN.blit(Assets.BACKGROUND, (x_pos_bg, y_pos_bg))
        SCREEN.blit(Assets.BACKGROUND, (image_width + x_pos_bg, y_pos_bg))
        
        if x_pos_bg <= -image_width:
            x_pos_bg = 0
        x_pos_bg -= game_speed
        
        # Draw UI
        text_points = Assets.FONT.render(f"Points: {points}", True, (0, 0, 0))
        text_speed = Assets.FONT.render(f"Speed: {game_speed}", True, (0, 0, 0))
        SCREEN.blit(text_points, (950, 50))
        SCREEN.blit(text_speed, (950, 80))
        
        pygame.display.update()
        clock.tick(FPS)
    
    pygame.quit()

if __name__ == '__main__':
    test_best_dino()