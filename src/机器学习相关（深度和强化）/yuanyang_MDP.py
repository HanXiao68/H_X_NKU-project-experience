import os
import pygame
from pygame.locals import *

class Yuanyang:
    def __init__(self):
        self.screen = None
        self.SCREEN_SIZE = (400, 300) # 窗口大小
        self.background = None
        self.bird1 = None
        self.bird2 = None
        self.obstacle = None
        # 障碍物砖块左上角点的横坐标
        self.obstacle_x = [120,240]*12   
        # 障碍物砖块左上角点的纵坐标
        self.obstacle_y = [0]*2+[20]*2+[40]*2+[60]*2+[80]*2+[100]*2+[180,120]+[200]*2+[220]*2+[240]*2+[260]*2+[280]*2   

    def render(self, close=False):
        pygame.init()
        self.screen = pygame.display.set_mode(self.SCREEN_SIZE, 0, 32) # 创建一个宽400高300的窗口，位深度为32
        pygame.display.set_caption("Window to " + "鸳鸯") # 设置窗口的标题
        
        # 加载需要的图片
        background_image_filename = 'resources/background.png'
        bird_image_filename = 'resources/bird.png'
        obstacle_image_filename = 'resources/obstacle.png'
        # 在窗口上加载背景
        self.background = pygame.image.load(background_image_filename).convert()
        screen_width, screen_height = self.SCREEN_SIZE
        self.screen.blit(self.background, (0, 0))
        # 在窗口上加载鸳鸯1
        self.bird1 = pygame.image.load(bird_image_filename).convert()
        self.screen.blit(self.bird1, (0, 0))
        # 在窗口上加载鸳鸯2
        self.bird2 = pygame.image.load(bird_image_filename).convert()
        self.screen.blit(self.bird2, (360, 0))
        # 在窗口上加载障碍物
        self.obstacle = pygame.image.load(obstacle_image_filename).convert()
        for i in range(len(self.obstacle_x)):
            self.screen.blit(self.obstacle, (self.obstacle_x[i], self.obstacle_y[i]))
        pygame.display.update()

if __name__ == "__main__":
    bird = Yuanyang()

    bird.render()

    while True:
        for event in pygame.event.get():
            if event.type == QUIT:
                pygame.quit()
                exit()