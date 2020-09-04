import os
import pygame
from pygame.locals import *
import time
import random
import numpy as np

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

        # 初始化状态空间，由上至下，从左到右，分别为1-100，宽为40，高为30
        self.states = [i for i in range(1,101)]
        # 当前状态
        self.state_c = None
        # 开始状态
        self.state_start = 1
        # 终结状态
        self.states_obstacle = [4,14,24,34,64,74,84,94,7,17,27,37,47,67,77,87,97]
        self.states_end = [10]
        self.states_terminal = self.states_obstacle + self.states_end
        # 动作空间
        self.actions = ['up','down','left','right']

        # 折扣因子
        self.gama = 0.8

        # 初始化每个状态可以执行的动作、每个状态当前决策的动作、q(s,a),returns(s,a),N
        self.state_action_init()
        self.state_policy_action_init()
        self.q_returns_N_init()

    # 把当前位置放置到起点位置
    def reset(self):
        self.state_c = self.state_start

    # 状态转换为坐标
    def state_to_xy(self,state):
        x = (state - 1) // 10
        y = (state - 1) % 10
        return x,y
    # 坐标转换为状态
    def xy_to_state(self,x,y):
        state = x * 10 + y +1
        return state

    # 初始化每个状态可以执行的动作
    def state_action_init(self):
        # 每个状态可以执行的动作
        self.states_actions = dict()
        for state in self.states:
            self.states_actions[state] = []
            if state in self.states_terminal:
                continue
            self.states_actions[state].append("up")
            self.states_actions[state].append("down")
            self.states_actions[state].append("left")
            self.states_actions[state].append("right")

    # 初始化每个状态当前决策的动作
    def state_policy_action_init(self):
        # 每个状态当前决策的动作
        self.states_policy_action = dict()
        for state in self.states:
            if state in self.states_terminal:
                self.states_policy_action[state] = None
            else:
                self.states_policy_action[state] = random.sample(self.states_actions[state],1)[0]

    # 初始化q(s,a),returns(s,a),N
    def q_returns_N_init(self):
        self.q_state_action = dict()
        self.returns_state_action = dict()
        self.N = dict()

        for state in self.states:
            if state in self.states_terminal:
                continue
            for action in self.states_actions[state]:
                self.q_state_action[str(state)+action] = 0.0
                self.returns_state_action[str(state)+action] = 0.0
                self.N[str(state)+action] = 0

    # 向下执行一步action动作
    def step(self,action):
        current_state = self.state_c
        if current_state in self.states_terminal:
            return current_state,0,True
        x,y = self.state_to_xy(current_state)
        if action == 'up':
            x = x - 1
        if action == 'down':
            x = x + 1
        if action == 'left':
            y = y - 1
        if action == 'right':
            y = y + 1

        if x < 0 or x > 9 or y < 0 or y > 9:
            return None, -10, 3            # 出界
        
        next_state = self.xy_to_state(x,y)
        self.state_c = next_state
        if next_state in self.states_end:  # 到达终点
            return next_state, 10, 0
        elif next_state in self.states_obstacle:    # 碰到障碍物 
            return next_state, -10, 1
        else:                         # 正常移动
            return next_state, 0, 2
    
    # 获得随机的s0 a0
    def get_random_state_action(self):
        while(True):
            state = random.randint(1,100)
            if state not in self.states_terminal:
                return state, random.sample(['up','down','right','left'],1)[0]
    
    # 生成一个episode
    def generate_episode(self, state_0, action_0):
        self.state_c = state_0
        # 返回下一个状态，即时回报，[正常移动,撞到障碍物,到达终点,出界]其中之一
        next_state, reward, flag = self.step(action_0)   
        episode = {0: [state_0, action_0, reward]}  # 字典类型
        
        i = 1
        while(flag == 2):
            # 贪婪策略选择下一个行为
            next_action = self.states_policy_action[next_state]   
            state_temp = next_state
            self.state_c = next_state
            # 获得即时回报，放入字典序列中，获得下一个状态，进入下个循环
            next_state, reward, flag = self.step(next_action)  
            episode[i] = [state_temp, next_action, reward]
            i += 1
            if i >= 50:   # 防止episode出现死循环
                break
        return episode
    
    # 探索初始化
    def MC_ES(self, episode):
        G = np.zeros(len(episode))
        for i in range(len(episode)):
            for j in range(len(episode)):
                if i + j < len(episode):
                    G[i] += pow(self.gama, j) * episode[i+j][2]
       
        state_action = []
        for i in range(len(G)):
            if([episode[i][0],episode[i][1]] in state_action):   # 判断是否第一次出现
                continue;
            state_action.append([episode[i][0],episode[i][1]])
            st = episode[i][0]
            at = episode[i][1]
            self.returns_state_action[str(st)+at] += G[i]
            self.N[str(st)+at] += 1
            self.q_state_action[str(st)+at] = self.returns_state_action[str(st)+at] / self.N[str(st)+at]

    def MC_improvement(self):
        for state in self.states:
            if state in self.states_terminal:
                continue
            # 初始化当前回报最大的行为
            max_value_action = self.states_actions[state][0]
            # 初始化当前最大回报
            max_value = self.q_state_action[str(state)+max_value_action] 
            for action in self.states_actions[state]:
                # 更新当前状态
                reward_q = self.q_state_action[str(state)+action]
                if reward_q > max_value:
                    max_value_action = action
                    max_value = reward_q
            self.states_policy_action[state] = max_value_action

    # 渲染整个场景
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
        # 在窗口上加载鸳鸯1的当前位置
        self.bird1 = pygame.image.load(bird_image_filename).convert()
        bird1_x,bird1_y = self.state_to_xy(self.state_c)
        print(bird1_x*30,bird1_y*40)
        self.screen.blit(self.bird1, (bird1_y*40,bird1_x*30))
        # 在窗口上加载鸳鸯2
        self.bird2 = pygame.image.load(bird_image_filename).convert()
        self.screen.blit(self.bird2, (360, 0))
        # 在窗口上加载障碍物
        self.obstacle = pygame.image.load(obstacle_image_filename).convert()
        for i in range(len(self.obstacle_x)):
            self.screen.blit(self.obstacle, (self.obstacle_x[i], self.obstacle_y[i]))
        pygame.display.update()

    def run(self):  
        for i in range(20000):
            # 随机选择初始状态和初始决策
            random_state,random_action = self.get_random_state_action()
            # print(random_state,random_action)
            # 生成一个episode
            episode = self.generate_episode(random_state,random_action)
            # print(episode)
            # 策略评估+改善
            self.MC_ES(episode)
            self.MC_improvement()

        # print(self.q_state_action)
        # print(self.states_policy_action)
        # 根据策略走一遍
        self.reset()
        while True:
            self.render()
            if self.states_policy_action[self.state_c] is not None:
                next_state, _, isdone = self.step(self.states_policy_action[self.state_c])
            else:
                isdone = 1
            if isdone == 1:
                self.render()
                print("succes")
                break
            time.sleep(0.2)
            
            # 可以随时退出
            for event in pygame.event.get():
                if event.type == QUIT:
                    pygame.quit()
                    exit()

if __name__ == "__main__":
    
    bird = Yuanyang()
    bird.reset()
    # bird.render()
    bird.run()