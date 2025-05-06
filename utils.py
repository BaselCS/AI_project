import math
import json
import os
import random

import pygame
from assets import Assets
from const import *
from gamestate import GameState
from obstacle import *


def remove_dinosaur(index,GameState):
    GameState.score_list[(GameState.dinosaurs[index].genome_id)%NUMBER_OF_POPULATION]= GameState.points
    GameState.dinosaurs[index].fitness -= 30
    GameState.dinosaurs.pop(index)
    if  len( GameState.gen_pool) > 0:
        GameState.gen_pool.pop(index)
        GameState.nets.pop(index)

def score(SCREEN,GameState):
    GameState.points += 1
    if GameState.points % 100 == 0:
        GameState.game_speed += 1
    text = Assets.FONT.render(f"Points: {GameState.points}", True, (0, 0, 0))
    SCREEN.blit(text, (950, 50))

def statistics(SCREEN,GameState):
    text_1 = Assets.FONT.render(f'Dinosaurs Alive: {len(GameState.dinosaurs)}', True, (0, 0, 0))
    text_2 = Assets.FONT.render(f'Generation: {GameState.current_generation}', True, (0, 0, 0))
    text_3 = Assets.FONT.render(f'Game Speed: {GameState.game_speed}', True, (0, 0, 0))

    SCREEN.blit(text_1, (50, 450))
    SCREEN.blit(text_2, (50, 480))
    SCREEN.blit(text_3, (50, 510))

def draw_background(SCREEN,GameState):
    image_width = Assets.BACKGROUND.get_width()
    SCREEN.blit(Assets.BACKGROUND, (GameState.x_pos_bg, GameState.y_pos_bg))
    SCREEN.blit(Assets.BACKGROUND, (image_width + GameState.x_pos_bg, GameState.y_pos_bg))
    
    if GameState.x_pos_bg <= -image_width:
        GameState.x_pos_bg = 0
    GameState.x_pos_bg -= GameState.game_speed

def spawn_obstacle(GameState):
    current_time = pygame.time.get_ticks()
    spawn_delay = GameState.spawn_cooldown - GameState.game_speed * 0.7 + random.randint(-50, 500)
    if len(GameState.obstacles)==0 or current_time - GameState.last_spawn_time > max(900, spawn_delay):
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
            GameState.obstacles.append(UpBird())
        GameState.last_spawn_time = current_time
        

def save_generation_data(generation, genomes):
    # Create save directory if it doesn't exist
    if not os.path.exists(SAVE_DIR):
        os.makedirs(SAVE_DIR)
    
    # Create generation directory
    gen_dir = os.path.join(SAVE_DIR, f"gen_{generation}")
    if not os.path.exists(gen_dir):
        os.makedirs(gen_dir)

    highest_fitness=-1
    highest_fitness_genome_id = -1    
    

    for genome_id, genome in genomes:
        if genome.fitness > highest_fitness:
            highest_fitness = genome.fitness
            highest_fitness_genome_id = genome_id
        dino_data = {
            'genome_id': genome_id,
            'fitness': genome.fitness,
            'score': GameState.score_list[genome_id%NUMBER_OF_POPULATION],
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
        
    # Save generation data
    with open(os.path.join(gen_dir, "generation_data.json"), 'w') as f:
        json.dump({
            'generation': generation,
            "highest_fitness": highest_fitness,
            'highest_fitness_genome_id': highest_fitness_genome_id,
            'score': GameState.score_list[highest_fitness_genome_id%NUMBER_OF_POPULATION],
        }, f, indent=2)




def handle_collisions(GameState,is_AI=True):
    to_remove = set()
    
    for obstacle in GameState.obstacles:
        for i, dinosaur in enumerate(GameState.dinosaurs):
            if dinosaur.rect.colliderect(obstacle.rect):    
                if is_AI:
                    GameState.gen_pool[i].fitness -= 5
                    to_remove.add(i)
                else:
                    return True
    
    # Remove dinosaurs that died
    for i in sorted(to_remove, reverse=True):
        remove_dinosaur(i,GameState)
        