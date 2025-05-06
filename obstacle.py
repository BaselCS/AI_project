import pygame
from assets import Assets
from const import *


class Obstacle:
    def __init__(self, image, number_of_cacti):
        self.image = image
        self.type = number_of_cacti
        self.rect = self.image[self.type].get_rect()
        self.rect.x = SCREEN_WIDTH

    def update(self,speed):
        self.rect.x -= speed
        return self.rect.x < -self.rect.width  # Returns True if off-SCREEN
    
    def draw(self, SCREEN):
        SCREEN.blit(self.image[self.type], self.rect)
        pygame.draw.rect(SCREEN, (255, 0, 0), self.rect, 2)  # Red rectangle

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

    def update(self,speed):
        self.rect.x -= speed
        # Animation - cycle through images
        self.index = (self.index + 1) % len(self.images)
        self.image = self.images[self.index]
        return self.rect.x < -self.rect.width  # Returns True if off-SCREEN    
            
    def draw(self, SCREEN):
        SCREEN.blit(self.image, (self.rect.x, self.rect.y))
        pygame.draw.rect(SCREEN, (0, 255, 0), self.rect, 2)  # Green rectangle
    
class UpBird(Bird):
    def __init__(self):
        super().__init__()
        self.rect.y = 150

class DownBird(Bird):
    def __init__(self):
        super().__init__()
        self.rect.y = 250
