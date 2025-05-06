import sys
import os
import pygame
import neat
import json
import argparse
from datetime import datetime
from assets import Assets
from const import DINO_Y_POS, FPS, SCREEN
from dinosaur import Dinosaur
from gamestate import GameState
from utils import draw_background, handle_collisions, score, spawn_obstacle, statistics

# Initialize pygame and load assets
pygame.init()
Assets.load()

def draw_fitness_value(SCREEN, fitness):
    """Display current fitness value on screen"""
    text = Assets.FONT.render(f"Fitness: {int(fitness)}", True, (0, 0, 0))
    SCREEN.blit(text, (50, 50))

def load_genome_from_file(genome_path, config):
    """Load a genome from a saved JSON file"""
    try:
        with open(genome_path, 'r') as f:
            genome_data = json.load(f)
        
        # Create a new genome object
        genome = neat.DefaultGenome(genome_data['genome_id'])
        
        # Set up nodes
        for node_data in genome_data['nodes']:
            node_id = node_data['id']
            genome.nodes[node_id] = neat.genome.DefaultNodeGene(node_id)
            genome.nodes[node_id].bias = node_data['bias']
            genome.nodes[node_id].response = node_data['response']
            genome.nodes[node_id].activation = node_data['activation']
            genome.nodes[node_id].aggregation = node_data['aggregation']
        
        # Set up connections
        for conn_data in genome_data['connections']:
            key = (conn_data['in'], conn_data['out'])
            genome.connections[key] = neat.genome.DefaultConnectionGene(key)
            genome.connections[key].weight = conn_data['weight']
            genome.connections[key].enabled = conn_data['enabled']
        
        return genome, genome_data['fitness'], genome_data['score']
    except Exception as e:
        print(f"Error loading genome: {e}")
        return None, 0, 0

def AI_play_single(dinosaur, net, obstacles, game_speed):
    """Handle AI decision making for a single dinosaur"""
    if obstacles:
        first = obstacles[0]
        second = obstacles[1] if len(obstacles) > 1 else first
        
        output = net.activate((
            dinosaur.rect.y,
            first.rect.x - dinosaur.rect.x,
            first.rect.height,
            first.rect.width,
            first.rect.y,
            second.rect.x - dinosaur.rect.x,
            second.rect.height,
            second.rect.width,
            second.rect.y,
            game_speed,
        ))
        
        if output[0] > 0.3 and dinosaur.rect.y == DINO_Y_POS:
            dinosaur.dino_jump = True
            dinosaur.dino_run = False
        if output[1] > 0.5 and not dinosaur.dino_jump:
            dinosaur.dino_crouch = True
        else:
            dinosaur.dino_crouch = False

def test_dino_game_loop(clock, dino, net):
    """Run game loop for testing a specific dinosaur"""
    GameState.reset()
    GameState.dinosaurs = [dino]
    current_fitness = 0
    
    while True:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                return -1
            
            # Allow manual control with spacebar to test interaction
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    pygame.quit()
                    return -1
        
        pygame.display.update()
        clock.tick(FPS)
        SCREEN.fill((255, 255, 255))
        
        # Update game objects
        sco = score(SCREEN, GameState)
        dino.update(sco)
        
        # Obstacle management
        spawn_obstacle(GameState)
        GameState.obstacles = [obstacle for obstacle in GameState.obstacles if not obstacle.update(GameState.game_speed)]
        
        # Handle collisions
        if handle_collisions(GameState, True):
            text = Assets.FONT.render("Game Over", True, (0, 0, 0))
            SCREEN.blit(text, (SCREEN.get_width() // 2 - text.get_width() // 2, 
                             SCREEN.get_height() // 2 - text.get_height() // 2))
            pygame.display.update()
            pygame.time.delay(2000)
            return GameState.points
        
        # AI decisions
        AI_play_single(dino, net, GameState.obstacles, GameState.game_speed)
        
        # Update fitness
        current_fitness += 0.2 + 0.2 * (GameState.game_speed / 100)
        
        # Drawing
        dino.draw(SCREEN, GameState.obstacles)
        for obstacle in GameState.obstacles:
            obstacle.draw(SCREEN)
            
        draw_fitness_value(SCREEN, current_fitness)
        statistics(SCREEN, GameState)
        score(SCREEN, GameState)
        draw_background(SCREEN, GameState)

def list_available_genomes(base_dir):
    """List all available genomes by generation"""
    available_gens = {}
    
    if not os.path.exists(base_dir):
        print(f"Save directory '{base_dir}' not found!")
        return available_gens
    
    # Get all generation directories
    for item in os.listdir(base_dir):
        gen_dir = os.path.join(base_dir, item)
        if os.path.isdir(gen_dir) and item.startswith("gen_"):
            gen_num = int(item.split("_")[1])
            
            # Read generation data
            gen_data_path = os.path.join(gen_dir, "generation_data.json")
            if os.path.exists(gen_data_path):
                try:
                    with open(gen_data_path, 'r') as f:
                        gen_data = json.load(f)
                        
                    best_genome_id = gen_data.get('highest_fitness_genome_id', -1)
                    best_fitness = gen_data.get('highest_fitness', 0)
                    best_score = gen_data.get('score', 0)
                    
                    # Get all dinosaur genomes in this generation
                    genomes = []
                    for file in os.listdir(gen_dir):
                        if file.startswith("dino_") and file.endswith(".json"):
                            genome_id = int(file.replace("dino_", "").replace(".json", ""))
                            genomes.append(genome_id)
                    
                    available_gens[gen_num] = {
                        'path': gen_dir,
                        'best_genome_id': best_genome_id,
                        'best_fitness': best_fitness,
                        'best_score': best_score,
                        'genomes': sorted(genomes)
                    }
                except Exception as e:
                    print(f"Error reading generation {gen_num}: {e}")
    
    return available_gens

def main():
    
    # Set up paths
    local_dir = os.path.dirname(__file__) if '__file__' in globals() else os.getcwd()
    config_path = os.path.join(local_dir, 'config.txt')
    
    # Load NEAT configuration
    try:
        config = neat.config.Config(
            neat.DefaultGenome,
            neat.DefaultReproduction,
            neat.DefaultSpeciesSet,
            neat.DefaultStagnation,
            config_path
        )
    except Exception as e:
        print(f"Error loading NEAT configuration: {e}")
        return
    
    genome_path = os.path.join(local_dir, 'Best_one.json')
    genome, fitness, saved_score = load_genome_from_file(genome_path, config)
    
    if genome is None:
        print("Failed to load genome!")
        return
    
    print(f"Loaded genome with fitness {fitness:.2f} and score {saved_score}")
    
    # Create neural network
    net = neat.nn.FeedForwardNetwork.create(genome, config)
    
    # Create dinosaur object
    dino = Dinosaur(genome_id=-1, genome=genome, config=config)
    
    # Run game loop
    clock = pygame.time.Clock()
    final_score = test_dino_game_loop(clock, dino, net)
    
    if final_score != -1:
        print(f"Test complete! Final score: {final_score}")

if __name__ == '__main__':
    main()
    
    
# خله يثبت الشاشة لو خسر
# نظف الأوامر
# سو شرح لتشغيل