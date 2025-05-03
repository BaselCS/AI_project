import glob
import logging
import pygame
import os
import random
import math
import neat
import json
from datetime import datetime

# Initialize pygame and load assets before anything else
pygame.init()

MAX_SIZE = 100 * 1024**3  # 100 GB
# Constants
SCREEN_WIDTH, SCREEN_HEIGHT = 1200, 900
FPS = 30
DINO_X_POS, DINO_Y_POS = 80, 310
JUMP_VELOCITY = 8.5
COUCHING_HEIGHT = 60
STAND_HEIGHT = 60
BACKGROUND_Y = 380
INITIAL_GAME_SPEED = 20
NUMBER_OF_GENERATIONS = 5000
SAVE_DIR = "dino_saves"

SCREEN = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))

class Assets:
    
    @staticmethod
    def load():
        try:
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
            Assets.Crouch={
                0: pygame.image.load(os.path.join("Assets/Dino", "DinoCrouch1.png")),
                1: pygame.image.load(os.path.join("Assets/Dino", "DinoCrouch2.png"))
            }
            Assets.LARGE_CACTUS = {
                0: pygame.image.load(os.path.join("Assets/Cactus", "LargeCactus1.png")),
                1: pygame.image.load(os.path.join("Assets/Cactus", "LargeCactus2.png")),
                2: pygame.image.load(os.path.join("Assets/Cactus", "LargeCactus3.png"))
            }
            Assets.BIRD = [
                pygame.image.load(os.path.join("Assets/Bird", "Bird1.png")),
                pygame.image.load(os.path.join("Assets/Bird", "Bird2.png"))  # Assuming there's a second frame
            ]        
            Assets.BACKGROUND = pygame.image.load(os.path.join("Assets/Other", "Track.png"))
            Assets.FONT = pygame.font.Font('freesansbold.ttf', 20)
        except pygame.error as e:
            print(f"Error loading assets: {e}")
            pygame.quit()
            exit()
            raise
        except FileNotFoundError as e:
            print(f"File not found: {e}")
            pygame.quit()
            exit()
            raise
        except Exception as e:
            print(f"Unexpected error: {e}")
            pygame.quit()
            exit()
            raise
        

# Load assets immediately when module is imported
Assets.load()

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
        
        self.invincible = False
        self.invincible_timer = 0
    
    def update(self):
        if self.invincible:
            self.invincible_timer -= 1
        if self.invincible_timer <= 0:
            self.invincible = False
        else:
            # Blink effect every 5 frames
            if self.invincible_timer % 5 == 0:
                self.blink_state = not self.blink_state
    
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
        
        
    def crouch(self):
        self.rect.height = self.crouching_height  # Shrink hitbox    
        self.image = Assets.Crouch[self.step_index %2]
        self.rect.x = DINO_X_POS
        self.rect.y = DINO_Y_POS + (self.standing_height - self.crouching_height)
        self.step_index += 1
        if self.step_index >= 10:
            self.step_index = 0
            self.dino_crouch = False
            self.dino_run = True
            self.rect.height = self.standing_height
            self.image = Assets.RUNNING[0]
            self.rect.y = DINO_Y_POS
                
        
    def draw(self, SCREEN):
        if not self.invincible or self.blink_state:
            # Draw the dinosaur sprite (standing or crouching)
            if self.dino_crouch:
                SCREEN.blit(Assets.Crouch[self.step_index // 5], (self.rect.x, self.rect.y + (self.standing_height - self.crouching_height)))
            else:
                SCREEN.blit(Assets.RUNNING[self.step_index // 5], (self.rect.x, self.rect.y))
            
            # Debug: Draw hitbox (red for crouching, green for standing)
            hitbox_color = (255, 0, 0) if self.dino_crouch else (0, 255, 0)
            pygame.draw.rect(SCREEN, hitbox_color, self.rect, 2)
            
            # Debug: Draw AI "vision" lines to obstacles
            if GameState.obstacles and hasattr(self, 'color'):  # Only if obstacles exist and color is defined
                for obstacle in GameState.obstacles:
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
                        obstacle.rect.center, 
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

class Bird():
    def __init__(self):
        self.images = Assets.BIRD  # Assuming this is a tuple of images
        self.image = self.images[0]  # Select the first image
        self.rect = self.image.get_rect()
        self.rect.x = SCREEN_WIDTH
        self.index = 0

    def update(self):
        self.rect.x -= GameState.game_speed
        # Animation - cycle through images
        self.index = (self.index + 1) % len(self.images)
        self.image = self.images[self.index]
        return self.rect.x < -self.rect.width  # Returns True if off-SCREEN    
            
    def draw(self, SCREEN):
        SCREEN.blit(self.image, (self.rect.x, self.rect.y))
    
class UpBird(Bird):
    def __init__(self):
        super().__init__()
        self.rect.y = 150


class DownBird(Bird):
    def __init__(self):
        super().__init__()
        self.rect.y = 225

  
def save_checkpoint(population, generation,config):
    neat.checkpoint.Checkpointer.save_checkpoint(
        filename=f"checkpoint-{generation}",
        population=population,
        generation=generation,
        config=config
    )
  

def handle_collisions():
    to_remove = set()
    
    for obstacle in GameState.obstacles:
        for i, dinosaur in enumerate(GameState.dinosaurs):
            if dinosaur.invincible:
                continue
            
            
            if dinosaur.rect.colliderect(obstacle.rect):
                # Apply penalty but don't immediately remove
                GameState.gen_pool[i].fitness -= 5
                dinosaur.hit_points =0
                dinosaur.invincible = True
                dinosaur.invincible_timer = 30  # ~1 second of invincibility
                dinosaur.blink_state = True
                
                if dinosaur.hit_points <= 0:
                    to_remove.add(i)
    
    # Remove dinosaurs that died
    for i in sorted(to_remove, reverse=True):
        remove_dinosaur(i)
        
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
    GameState.dinosaurs[index].fitness -= 30 
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
    if len(GameState.obstacles) < random.randint(1, 5) and current_time - GameState.last_spawn_time + random.randint(-50,500) > GameState.spawn_cooldown:
        obstacle= random.choice([SmallCactus, LargeCactus, UpBird, DownBird])
        if obstacle == SmallCactus:
            for _ in range(random.randint(1, 3)):
                GameState.obstacles.append(SmallCactus(Assets.SMALL_CACTUS, random.randint(0, 2)))
        elif obstacle == LargeCactus:
            for _ in range(random.randint(1, 3)):
                GameState.obstacles.append(LargeCactus(Assets.LARGE_CACTUS, random.randint(0, 2)))
        elif obstacle== UpBird:
            GameState.obstacles.append(UpBird())
        elif obstacle== DownBird:
            GameState.obstacles.append(DownBird())
        GameState.last_spawn_time = current_time
    
    

def delete_oldest_files(path):
    files = sorted(
        (f for f in glob.glob(f"{path}/*") if os.path.isfile(f)),
        key=lambda x: os.path.getctime(x)
    )
    while os.path.getsize(SAVE_DIR) > MAX_SIZE*(10/100) and files:
        os.remove(files.pop(0))


def save_generation_data(generation, genomes):
    # Create save directory if it doesn't exist
    if not os.path.exists(SAVE_DIR):
        os.makedirs(SAVE_DIR)
    elif os.path.getsize(SAVE_DIR) > MAX_SIZE:
            delete_oldest_files(SAVE_DIR)
        
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
        try:
            with open(os.path.join(gen_dir, f"dino_{genome_id}.json"), 'w') as f:
                json.dump(dino_data, f, indent=2)
        except IOError as e:
            print(f"Error saving dino {genome_id}: {e}")       
            
             
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
        dino = Dinosaur(genome_id=genome_id, genome=genome, config=config)
        GameState.dinosaurs.append(dino)
        GameState.gen_pool.append(genome)
        GameState.nets.append(neat.nn.FeedForwardNetwork.create(genome, config))
        genome.fitness = 0

    running = True
    try:
        while running:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                    
            pygame.display.update()
            clock.tick(FPS)
            if not GameState.dinosaurs:
                break

            SCREEN.fill((255, 255, 255))
            
            if GameState.current_generation % 30 == 3 :
                save_checkpoint(GameState.population, GameState.current_generation, config)
                print(f"Checkpoint saved for generation {GameState.current_generation}")
                
            
            
            # --- 1. Update All Game Objects First ---
            for dinosaur in GameState.dinosaurs:
                dinosaur.update()
            
            # Obstacle management
            spawn_obstacle()
            GameState.obstacles = [obstacle for obstacle in GameState.obstacles if not obstacle.update()]
            
            # --- 2. Handle Collisions (REPLACES your manual collision check) ---
            handle_collisions()  # This now handles ALL collision logic
            
            # --- 3. AI Decisions  ---
            for i, dinosaur in enumerate(GameState.dinosaurs):
                if GameState.obstacles:
                    closest_obstacle = GameState.obstacles[0]
                    output = GameState.nets[i].activate((
                        dinosaur.rect.y,
                        closest_obstacle.rect.x - dinosaur.rect.x,
                        closest_obstacle.rect.height,
                        closest_obstacle.rect.width,
                        GameState.game_speed,
                        closest_obstacle.rect.y
                    ))
                    
                    if output[0] > 0.5 and dinosaur.rect.y == DINO_Y_POS:
                        dinosaur.dino_jump = True
                        dinosaur.dino_run = False

                    if output[1] > 0.5 and not dinosaur.dino_jump:
                        dinosaur.dino_crouch = True
                    else:
                        dinosaur.dino_crouch = False
            
            # --- 4. Single Fitness Reward Point ---
            for i, dinosaur in enumerate(GameState.dinosaurs):
                if not dinosaur.invincible:
                    GameState.gen_pool[i].fitness += 0.2 + 0.2 * (GameState.game_speed / 100)  # Combined into one reward
            
            # --- 5. Drawing Phase ---
            for dinosaur in GameState.dinosaurs:
                dinosaur.draw(SCREEN)
                
            for obstacle in GameState.obstacles:
                obstacle.draw(SCREEN)
            
            statistics(SCREEN)
            score(SCREEN)
            draw_background(SCREEN)
        
            save_generation_data(GameState.current_generation, genomes)
    except Exception as e:
        print(f"Error during evaluation: {e}")
        save_generation_data(GameState.current_generation, genomes)
        # Clean up and exit
        pygame.quit()
        exit()
    finally:
        pygame.quit()
        exit()
        

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
    logging.basicConfig(filename='training.log', level=logging.INFO)
    local_dir = os.path.dirname(__file__) if '__file__' in globals() else os.getcwd()
    config_path = os.path.join(local_dir, 'config.txt')
    
    # Clear previous saves if needed (optional)
    if os.path.exists(SAVE_DIR):
        import shutil
        shutil.rmtree(SAVE_DIR)
    else :
        os.makedirs(SAVE_DIR)
    
    
    run(config_path)
    
    
    
    
    


#test code
    
def test_saved_genome(config, genome_path):
    genome = load_genome(genome_path, config)
    net = neat.nn.FeedForwardNetwork.create(genome, config)
    
    dino = Dinosaur(genome=genome, config=config)
    clock = pygame.time.Clock()
    running = True

    GameState.reset()
    GameState.dinosaurs = [dino]

    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

        SCREEN.fill((255, 255, 255))
        draw_background(SCREEN)
        dino.update()

        spawn_obstacle()
        GameState.obstacles = [o for o in GameState.obstacles if not o.update()]
        handle_collisions()

        if GameState.obstacles:
            obstacle = GameState.obstacles[0]
            output = net.activate((
                dino.rect.y,
                obstacle.rect.x - dino.rect.x,
                obstacle.rect.height,
                obstacle.rect.width,
                GameState.game_speed,
                obstacle.rect.y
            ))

            if output[0] > 0.5 and dino.rect.y == DINO_Y_POS:
                dino.dino_jump = True
                dino.dino_run = False

            if output[1] > 0.5 and not dino.dino_jump:
                dino.dino_crouch = True
            else:
                dino.dino_crouch = False

        for obstacle in GameState.obstacles:
            obstacle.draw(SCREEN)
        dino.draw(SCREEN)
        statistics(SCREEN)
        score(SCREEN)

        pygame.display.update()
        clock.tick(FPS)

        if not GameState.dinosaurs:
            break
    
def load_genome(filepath, config):
    with open(filepath, 'r') as f:
        data = json.load(f)

    # Create an empty genome with the correct ID and config
    genome = neat.DefaultGenome(data['genome_id'])
    genome.configure_new(config.genome_config)

    # Load node data
    for node_info in data['nodes']:
        node = genome.nodes[node_info['id']]
        node.bias = node_info['bias']
        node.activation = node_info['activation']
        node.aggregation = node_info['aggregation']
        node.response = node_info['response']

    # Load connections
    genome.connections.clear()
    for conn in data['connections']:
        key = (conn['in'], conn['out'])
        connection = neat.DefaultConnectionGene(key)
        connection.weight = conn['weight']
        connection.enabled = conn['enabled']
        genome.connections[key] = connection

    return genome
