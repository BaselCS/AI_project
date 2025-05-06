import sys
import pygame
import os
import neat
from datetime import datetime

from assets import Assets
from const import DINO_Y_POS, FPS, NUMBER_OF_GENERATIONS, NUMBER_OF_POPULATION, SCREEN
from dinosaur import Dinosaur
from file_manger import move_and_rename
from gamestate import GameState
from utils import draw_background, handle_collisions, save_generation_data, score, spawn_obstacle, statistics

#أقرا readme.md


# Initialize pygame and load assets before anything else
pygame.init()

Assets.load()

def draw_fitness_value(SCREEN):
    fitness=[]
    
    for  gen in GameState.gen_pool:
        fitness.append(gen.fitness)
    
    if fitness:
        # Get top 10 fitness scores sorted by fitness (descending order)
        top_fitness = sorted(fitness, reverse=True)[:10]
        
        # Table header
        header = Assets.FONT.render("Top 10 Fitness Scores:", True, (0, 0, 0))
        SCREEN.blit(header, (50, 50))
        
        # Display each dinosaur's fitness in a horizontal table
        for i,fitness in enumerate(top_fitness):
            text = Assets.FONT.render(f"{int(fitness)}", True, (0, 0, 0))
            SCREEN.blit(text, (50 + i * 100, 80))  # Adjust x position for each column
    else:
        text = Assets.FONT.render(f"Fitness: {GameState.points}", True, (0, 0, 0))
        SCREEN.blit(text, (50, 50))


def manual_play(self):
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            self.running = False

        if event.type == pygame.KEYDOWN:
            if event.key == pygame.K_UP:
                self.dino_jump = True
                self.dino_run = False
            elif event.key == pygame.K_DOWN:
                self.dino_crouch = True
                self.dino_run = False
                
                
        if event.type == pygame.KEYUP:
            if event.key == pygame.K_DOWN:
                self.dino_crouch = False
                self.dino_run = True
                
def AI_Play():
  for i, dinosaur in enumerate(GameState.dinosaurs):
    if GameState.obstacles:
        first = GameState.obstacles[0]
        second = GameState.obstacles[1] if len(GameState.obstacles) > 1 else first  # تجنب الخطأ

        output = GameState.nets[i].activate((
            dinosaur.rect.y,
            first.rect.x - dinosaur.rect.x,
            first.rect.height,
            first.rect.width,
            first.rect.y,
            second.rect.x - dinosaur.rect.x,
            second.rect.height,
            second.rect.width,
            second.rect.y,
            GameState.game_speed,
        ))

        if output[0] > 0.3 and dinosaur.rect.y == DINO_Y_POS:
            dinosaur.dino_jump = True
            dinosaur.dino_run = False

        if output[1] > 0.5 and not dinosaur.dino_jump:
            dinosaur.dino_crouch = True
        else:
            dinosaur.dino_crouch = False

    for i, dinosaur in enumerate(GameState.dinosaurs):
        GameState.gen_pool[i].fitness += 0.2 + 0.2 * (GameState.game_speed / 100)


def Game_loop(clock,is_AI=True):
    if not is_AI:
        GameState.dinosaurs = [Dinosaur()]

    while True:  
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                return -1
          
        pygame.display.update()
        clock.tick(FPS)
        if not GameState.dinosaurs:
            return

        SCREEN.fill((255, 255, 255))
        
        # --- 1. Update All Game Objects First ---
        sco = score(SCREEN,GameState)
        for dinosaur in GameState.dinosaurs:
            dinosaur.update(sco)
        
        
        # Obstacle management
        spawn_obstacle(GameState)
        GameState.obstacles = [obstacle for obstacle in GameState.obstacles if not obstacle.update(GameState.game_speed)]
        
        # --- 2. Handle Collisions (REPLACES your manual collision check) ---
        if handle_collisions(GameState,is_AI) :
 
            # بيدخل هنا لو البشري يتحكم و صقع
            text = Assets.FONT.render("Game Over", True, (0, 0, 0))
            SCREEN.blit(text, (SCREEN.get_width() // 2 - text.get_width() // 2, SCREEN.get_height() // 2 - text.get_height() // 2))
            pygame.display.update()
            pygame.time.delay(2000)
            return -1
            
        
        # --- 3. AI Decisions  ---
        if is_AI:
            AI_Play()
        else:
            manual_play(GameState.dinosaurs[0])
            
        # --- 5. Drawing Phase ---
        for dinosaur in GameState.dinosaurs:
            dinosaur.draw(SCREEN, GameState.obstacles)
        
            
        for obstacle in GameState.obstacles:
            obstacle.draw(SCREEN)
                
        draw_fitness_value(SCREEN)
        statistics(SCREEN,GameState)
        score(SCREEN,GameState)
        draw_background(SCREEN,GameState)
    

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
    
    GameState.score_list=[0]*NUMBER_OF_POPULATION
    
    value=Game_loop(clock)

    if value == -1:
        save_generation_data(GameState.current_generation, genomes)
        sys.exit(0)
    
    
    
    save_generation_data(GameState.current_generation, genomes)
    
    if GameState.current_generation >= NUMBER_OF_GENERATIONS: 
        return



def run(config_path):
    #100GB
    MAXSIZE=100 * 1024 * 1024 * 1024  # 100 GB

    while True:
        if not os.path.exists(OLD_DIR):
            os.makedirs(OLD_DIR)
            
        if os.path.getsize(OLD_DIR) > MAXSIZE:
            print("Old directory size exceeded 100GB.")
            break
        
        # Initialize GameState
        GameState.reset()
        # Clear previous saves if needed
        if os.path.exists(SAVE_DIR):
            move_and_rename(OLD_DIR,SAVE_DIR)
        else:
            os.makedirs(SAVE_DIR)
        
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
        GameState.restart()
        print("#"*900)



if __name__ == '__main__':
    #أقرا readme.md
    local_dir = os.path.dirname(__file__) if '__file__' in globals() else os.getcwd()
    config_path = os.path.join(local_dir, 'config.txt')
    
    SAVE_DIR = os.path.join(local_dir, 'dino_saves') 
    OLD_DIR = os.path.join(local_dir, 'old_saves')  
   

    run(config_path)
    # Game_loop(pygame.time.Clock(),is_AI=False)    
    