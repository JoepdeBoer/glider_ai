import time
import gymnasium as gym
import numpy as np
import random
import os
import pygame

from custom_gym.rendering import field_to_rgb, draw_plane, subsurface

class CompEnv(gym.Env):
    metadata = {"render_modes": ["human", "training"], "render_fps": 10}
    def __init__(self, agent, wind, time_limit, render_mode = None):
        #super(competttitionEnv, self).__init__() # allowing to pass parameters
        self.plane = agent
        self.wind = wind
        self.time_limit = time_limit
        self.timestep = 0
        self.done = False
        self.action_space = gym.spaces.Box(low=-1, high=1, shape=(2,), dtype=np.float32)
        self.observation_space = gym.spaces.Dict({'xyz': gym.spaces.Box(low=-np.inf, high=np.inf, shape=(3,), dtype=np.float32),
                                                  'V': gym.spaces.Box(low=agent.vstall, high=agent.vne, dtype=np.float32),
                                                  'thetha': gym.spaces.Box(low=-np.pi, high=np.pi, dtype=np.float32),
                                                  'b': gym.spaces.Box(low=-np.pi/3, high=np.pi/3, dtype=np.float32),
                                                  'updraft': gym.spaces.Box(low=-10, high=10, dtype=np.float32),
                                                  'objective': gym.spaces.Box(low=-np.inf, high=np.inf, shape=(2,), dtype=np.float32)}) #For multi agent implement each others state in this dict
        # rendering
        assert render_mode is None or render_mode in self.metadata["render_modes"]
        self.render_mode = render_mode
        self.display_size = 1000
        self.display = None
        self.updraft_rgb = None
        self.clock = None
        self.font = None

    def step(self, action):
        self.timestep += 1
        action = action.reshape((2,)) # inconsistent shape is sometimes returned really anoying
        self.plane.take_action(action)
        self.plane.move(self.wind)
        state = self.plane.get_state(self.wind) # new state
        reward, self.done = self.reward()
        info = {} # not used

        if (self.render_mode == 'human' or self.render_mode == 'training') and not self.done:
            self._render_frame()

        return state, reward, self.done, False, info

    def reward(self):
        distance = np.sqrt(
            (self.plane.x - self.plane.objective[0]) ** 2 + (self.plane.y - self.plane.objective[1]) ** 2)
        if self.timestep < 2:
            reward = 0
        else:
            prev_dist = np.sqrt((self.plane.history[-2][0] - self.plane.objective[0])**2 + (self.plane.history[-2][1] - self.plane.objective[1])**2)
            delta_dist = prev_dist - distance # distance travelled to objective
            delta_alt = self.plane.z - self.plane.history[-2][2] # altitude gain

            #Base reward function
            reward = delta_dist/1000 # reward is representative of distance travelled to objective
        done = True

        #Sparse reward function
        if self.plane.V < 1.2 * self.plane.vstall or self.plane.V > self.plane.vne:
            reward = -1000 - distance/1000
            print('velocity out of bounds')
        elif self.plane.b > np.pi/3 or self.plane.b < -np.pi/3:
            reward = -1000 - distance/1000
            print('bank angle to large')
        elif self.plane.z <= 0:
            reward = -distance/1000
            print(f'plane crashed {self.plane.z}')
        elif self.timestep >= self.time_limit:
            reward = -distance/1000
            print('time limit reached')
        elif distance < 2000: # if plane hit the objective
            reward = 1000 + (self.time_limit - self.timestep)
            print('objective reached')
        else:
            done = False

        return reward, done


    def reset(self, seed=None, options=None):
        #TODO implement logging of results
        self.done = False
        self.timestep = 0

        #selecting a random windfield
        self.wind.load_field() # Load one of the random windfields

        # # resetting agent location to middle of the field
        self.plane.x = self.wind.field.shape[0]/2 * self.wind.resolution
        self.plane.y = self.wind.field.shape[1]/2 * self.wind.resolution

        self.plane.z = 1000  # lower than distance to objective/L/D_max such that utilising thermals is required
        self.plane.V = 30
        self.plane.b = 0
        self.plane.theta = random.uniform(-np.pi, np.pi)
        self.plane.history = []
        # setting the objective to a location 50km away
        r = 50e3
        theta = random.uniform(0, 2*np.pi)
        self.plane.objective = np.array([self.plane.x + r*np.cos(theta), self.plane.y + r*np.sin(theta)]).reshape((2,))

        # get starting state
        state = self.plane.get_state(self.wind)  # new state

        if self.render_mode == "human":
            self._render_frame()

        info = {} # not used
        return state, info

    def render(self):
        if self.render_mode:
            return self._render_frame()

    # def render(self):
    #     self.display.fill('blue')
    #     # self.display.blit(subsurface(self.updraft_rgb, self.plane.x, self.plane.y,
    #     #                      WINDFIELD_resolution, WIDTH, HEIGHT), (0, 0))
    #     # # draw_plane(self.plane, self.display, self.font)
    #     pygame.event.pump()
    #     pygame.display.update()
    #     self.clock.tick(FPS)

    
    def _render_frame(self):
        if self.render_mode:
            if pygame.get_init() is False:
                pygame.init()
                pygame.display.set_caption('Flying High')
            if self.display is None:
                self.display = pygame.display.set_mode((self.display_size, self.display_size))
            if self.font is None:
                self.font = pygame.font.SysFont('Courier New', 30)
            if self.clock is None:
                self.clock = pygame.time.Clock()
            if self.updraft_rgb is None:
                self.updraft_rgb = field_to_rgb(self.wind.field)

        # background creation

        surface = subsurface(self.updraft_rgb, self.plane.x, self.plane.y,
                             self.wind.resolution, self.display_size, self.display_size)
        draw_plane(self.plane, surface, self.font, self.wind)
        self.display.blit(surface, surface.get_rect())
        pygame.event.pump()
        pygame.display.update()

        if self.render_mode == "human":
            # The following line will automatically add a delay to keep the framerate stable.
            self.clock.tick(self.metadata["render_fps"])


    #
    def close(self):
        if self.render_mode == "human":
            self.plane.plot_history()
        if self.display is not None:
            self.display = None
            self.clock = None
            self.font = None
            self.updraft_rgb = None
            pygame.display.quit()
            pygame.quit()



if __name__ == '__main__':
    print('nothing here')
