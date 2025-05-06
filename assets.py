import pygame
import os


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
        