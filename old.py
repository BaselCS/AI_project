import pygame
import os
import random
import math
import sys
import neat
import json
from datetime import datetime

# Initialize pygame and load assets before anything else
pygame.init()

# Constants
SCREEN_WIDTH, SCREEN_HEIGHT = 1200, 900
FPS = 30
DINO_X_POS, DINO_Y_POS = 80, 310
JUMP_VELOCITY = 8.5
BACKGROUND_Y = 380
INITIAL_GAME_SPEED = 20
NUMBER_OF_GENERATIONS = 500
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

# Load assets immediately when module is imported
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
        self.net = neat.nn.FeedForwardNetwork.create(genome, config) if genome and config else None
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
        pygame.draw.rect(SCREEN, self.color, self.rect, 2)
        
        for obstacle in GameState.obstacles:
            pygame.draw.line(SCREEN, self.color, 
                           (self.rect.x + 54, self.rect.y + 12),
                           obstacle.rect.center, 2)
    
    def to_dict(self):
        return {
            'genome_id': self.genome_id,
            'color': self.color,
            'fitness': self.fitness,
            'jump_velocity': self.jump_vel,
            'position': (self.rect.x, self.rect.y)
        }

class Obstacle:
    def __init__(self, image, number_of_cacti):
        self.image = image
        self.type = number_of_cacti
        self.rect = self.image[self.type].get_rect()
        self.rect.x = SCREEN_WIDTH

    def update(self):
        self.rect.x -= GameState.game_speed
        return self.rect.x < -self.rect.width  # Returns True if off-SCREEN
    
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

class GameState:
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
    current_generation = 0
    best_dinos = {}  # To store best dinos from each generation

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

def score(SCREEN):
    GameState.points += 1
    if GameState.points % 100 == 0:
        GameState.game_speed += 1
    text = Assets.FONT.render(f"Points: {GameState.points}", True, (0, 0, 0))
    SCREEN.blit(text, (950, 50))

def statistics(SCREEN):
    text_1 = Assets.FONT.render(f'Dinosaurs Alive: {len(GameState.dinosaurs)}', True, (0, 0, 0))
    text_2 = Assets.FONT.render(f'Generation: {GameState.current_generation}', True, (0, 0, 0))
    text_3 = Assets.FONT.render(f'Game Speed: {GameState.game_speed}', True, (0, 0, 0))

    SCREEN.blit(text_1, (50, 450))
    SCREEN.blit(text_2, (50, 480))
    SCREEN.blit(text_3, (50, 510))

def draw_background(SCREEN):
    image_width = Assets.BACKGROUND.get_width()
    SCREEN.blit(Assets.BACKGROUND, (GameState.x_pos_bg, GameState.y_pos_bg))
    SCREEN.blit(Assets.BACKGROUND, (image_width + GameState.x_pos_bg, GameState.y_pos_bg))
    
    if GameState.x_pos_bg <= -image_width:
        GameState.x_pos_bg = 0
    GameState.x_pos_bg -= GameState.game_speed

def spawn_obstacle():
    current_time = pygame.time.get_ticks()
    if len(GameState.obstacles) < 3 and current_time - GameState.last_spawn_time+random.randint(0,1500)  > GameState.spawn_cooldown:
        if random.randint(0, 1) == 0:
            for _ in range(random.randint(1, 3)):
                GameState.obstacles.append(SmallCactus(Assets.SMALL_CACTUS, random.randint(0, 2)))
        else:
            for _ in range(random.randint(1, 3)):
                GameState.obstacles.append(LargeCactus(Assets.LARGE_CACTUS, random.randint(0, 2)))
        GameState.last_spawn_time = current_time

def save_generation_data(generation, genomes):
    # Create save directory if it doesn't exist
    if not os.path.exists(SAVE_DIR):
        os.makedirs(SAVE_DIR)
    
    # Create generation directory
    gen_dir = os.path.join(SAVE_DIR, f"gen_{generation}")
    if not os.path.exists(gen_dir):
        os.makedirs(gen_dir)
    
    # Save each dinosaur's data
    best_fitness = -1
    best_dino = None
    
    for genome_id, genome in genomes:
        dino_data = {
            'genome_id': genome_id,
            'fitness': genome.fitness,
            'connections': [
                {
                    'in': c.key[0],
                    'out': c.key[1],
                    'weight': c.weight,
                    'enabled': c.enabled
                } 
                for c in genome.connections.values()
            ],
            'nodes': [
                {
                    'id': node_id,
                    'bias': node.bias,
                    'activation': node.activation,
                    'aggregation': node.aggregation,
                    'response': node.response
                }
                for node_id, node in genome.nodes.items()
            ]
        }
        
        # Save to file
        with open(os.path.join(gen_dir, f"dino_{genome_id}.json"), 'w') as f:
            json.dump(dino_data, f, indent=2)
        
        # Track best dino
        if genome.fitness > best_fitness:
            best_fitness = genome.fitness
            best_dino = dino_data
    
    # Save best dino of this generation
    if best_dino:
        GameState.best_dinos[generation] = best_dino
        with open(os.path.join(gen_dir, "best_dino.json"), 'w') as f:
            json.dump(best_dino, f, indent=2)
    
    # Save summary of all generations
    with open(os.path.join(SAVE_DIR, "generations_summary.json"), 'w') as f:
        json.dump(GameState.best_dinos, f, indent=2)
def eval_genomes(genomes, config):
    GameState.reset()
    clock = pygame.time.Clock()
    
    GameState.current_generation += 1
    print(f"\n--- Starting Generation {GameState.current_generation} ---")
    
    # Initialize NEAT population
    for genome_id, genome in genomes:
        dino = Dinosaur()
        dino.genome_id = genome_id
        GameState.dinosaurs.append(dino)
        GameState.gen_pool.append(genome)
        GameState.nets.append(neat.nn.FeedForwardNetwork.create(genome, config))
        genome.fitness = 0

    while True:
        if not GameState.dinosaurs:
            break

        SCREEN.fill((255, 255, 255))
        
        # Game logic
        for dinosaur in GameState.dinosaurs:
            dinosaur.update()
            dinosaur.draw(SCREEN)
        
        if not GameState.dinosaurs:
            break
        
        # Obstacle management
        spawn_obstacle()
        GameState.obstacles = [obstacle for obstacle in GameState.obstacles if not obstacle.update()]
        
        for obstacle in GameState.obstacles:
            obstacle.draw(SCREEN)
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
        
        # Update fitness for surviving dinosaurs
        for i, dinosaur in enumerate(GameState.dinosaurs):
            GameState.gen_pool[i].fitness += 0.1
        
        # Drawing
        statistics(SCREEN)
        score(SCREEN)
        draw_background(SCREEN)
        
        pygame.display.update()
        clock.tick(FPS)
    
    # Save generation data after simulation ends
    save_generation_data(GameState.current_generation, genomes)

def run(config_path):
    config = neat.config.Config(
        neat.DefaultGenome,
        neat.DefaultReproduction,
        neat.DefaultSpeciesSet,
        neat.DefaultStagnation,
        config_path
    )
    
    # Add reporter to show progress in console
    stats = neat.StatisticsReporter()
    GameState.population = neat.Population(config)
    GameState.population.add_reporter(stats)
    GameState.population.add_reporter(neat.StdOutReporter(True))
    
    # Run for up to NUMBER_OF_GENERATIONS generations
    GameState.population.run(eval_genomes, NUMBER_OF_GENERATIONS)
    
    # After all generations, print summary
    print("\n--- Training Complete ---")
    print(f"Saved data for {GameState.current_generation} generations in '{SAVE_DIR}' directory")
    
    # Find the overall best dinosaur
    if GameState.best_dinos:
        best_gen = max(GameState.best_dinos.items(), key=lambda x: x[1]['fitness'])
        print(f"\nBest dinosaur was from generation {best_gen[0]} with fitness {best_gen[1]['fitness']}")
        print(f"You can find its data in: {os.path.join(SAVE_DIR, f'gen_{best_gen[0]}', 'best_dino.json')}")

if __name__ == '__main__':
    local_dir = os.path.dirname(__file__)
    config_path = os.path.join(local_dir, 'config.txt')
    
    # Clear previous saves if needed (optional)
    if os.path.exists(SAVE_DIR):
        print(f"Warning: '{SAVE_DIR}' directory already exists. Previous data will be overwritten.")
    
    run(config_path)