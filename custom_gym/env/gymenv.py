import time
import gym
import numpy as np
import random
import os
import pygame

from custom_gym.rendering import field_to_rgb, draw_plane

WIDTH, HEIGHT = 500, 500 # screen width and height
WINDFIELD_resolution = 10
FPS = 24


class CompEnv(gym.Env):
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 24}
    def __init__(self, agent, wind, time_limit, render_mode = None):
        #super(competttitionEnv, self).__init__() # allowing to pass parameters
        self.plane = agent
        self.wind = wind
        self.time_limit = time_limit
        self.timestep = 0
        self.done = False
        self.action_space = gym.spaces.Dict({'db': gym.spaces.Box(low=-1, high=1, shape=(1,), dtype=np.float32),
                                             'dV': gym.spaces.Box(low=-1, high=1, shape=(1,), dtype=np.float32)}) # need to specify two actions possible gym.spaces.Box(low=-1, high=1, shape=(2,), dtype=np.float32) # need to specify two actions possible
        self.observation_space = gym.spaces.Dict({'xyz': gym.spaces.Box(low=-np.inf, high=np.inf, shape=(3,), dtype=np.float32),
                                                  'V': gym.spaces.Box(low=agent.vstall, high=agent.vne, dtype=np.float32),
                                                  'thetha': gym.spaces.Box(low=-np.pi, high=np.pi, dtype=np.float32),
                                                  'b': gym.spaces.Box(low=-np.pi/3, high=np.pi/3, dtype=np.float32),
                                                  'updraft': gym.spaces.Box(low=-10, high=10, dtype=np.float32),
                                                  'objective': gym.spaces.Box(low=-np.inf, high=np.inf, shape=(2,), dtype=np.float32)}) #For multi agent implement each others state in this dict
        # rendering
        assert render_mode is None or render_mode in self.metadata["render_modes"]
        self.render_mode = render_mode
        self.display_size = 800
        self.display = None
        self.updraft_rgb = None
        self.clock = None
        self.font = None

    def step(self, action):
        self.timestep += 1
        self.plane.take_action(action['db'], action['dV'])
        self.plane.move(self.wind)
        state = self.plane.get_state(self.wind) # new state
        reward, self.done = self.reward()
        info = {} # not used

        if self.render_mode == 'human' and not self.done:
            self._render_frame()

        return state, reward, self.done, False, info

    def reward(self):
        distance = np.sqrt((self.plane.x - self.plane.objective[0])**2 + (self.plane.y - self.plane.objective[1])**2)
        alt = self.plane.z

        #Base reward function
        reward = -distance + alt*self.plane.bestLD - self.timestep
        done = True
        if self.plane.z <= 0:
            reward -= 500 * distance
        elif self.timestep >= self.time_limit:
            reward -= 500 * distance
        elif distance < 2000: # if plane hit the objective
            reward += 1000 * (self.time_limit-self.timestep)
        else:
            done = False
        return reward, done


    def reset(self, seed=None, options=None):
        #TODO implement logging of results
        self.done = False
        self.timestep = 0

        #selecting a random windfield
        self.wind.load_field() # Load one of the random windfields

        # resetting agent location to middle of the field
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
        self.plane.objective = np.array([self.plane.x + r*np.cos(theta), self.plane.y + r*np.sin(theta)])

        # get starting state
        state = self.plane.get_state(self.wind)  # new state

        if self.render_mode == "human":
            self._render_frame()

        info = {} # not used
        return state, info

    def render(self):
        if self.render_mode == "rgb_array":
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
        if self.display is None and self.render_mode == "human":
            pygame.init()
            pygame.display.init()
            self.display = pygame.display.set_mode((self.display_size, self.display_size))
            self.font = pygame.font.SysFont('Courier New', 30)
        if self.clock is None and self.render_mode == "human":
            self.clock = pygame.time.Clock()
        if self.font is None:
            self.font = pygame.font.SysFont('Courier New', 30)
        if self.updraft_rgb is None:
            self.updraft_rgb = field_to_rgb(self.wind.field)

        # background creation
        # TODO when plane is out of the field make it a color
        x1 = y1 = int((self.plane.x - self.display_size / 2) / self.wind.resolution)
        x2 = y2 = int(x1 + round(self.display_size / self.wind.resolution))
        localmap = self.updraft_rgb[x1:x2, y1:y2, :] # TODO x1, y1, x2, y2 can be out of range!!!
        size = localmap.shape[1::-1] # TODO test
        surface = pygame.image.frombuffer(localmap.flatten(), size, 'RGB')
        surface = pygame.transform.scale_by(surface, self.wind.resolution)
        # surface = pygame.Surface((self.display_size, self.display_size))
        # surface.fill((255, 255, 255))
        draw_plane(self.plane, surface, self.font)

        if self.render_mode == "human":
            # The following line copies our drawings from `canvas` to the visible window
            self.display.blit(surface, surface.get_rect())
            pygame.event.pump()
            pygame.display.update()

            # We need to ensure that human-rendering occurs at the predefined framerate.
            # The following line will automatically add a delay to keep the framerate stable.
            self.clock.tick(self.metadata["render_fps"])
        else:  # rgb_array
            return np.transpose(
                np.array(pygame.surfarray.pixels3d(surface)), axes=(1, 0, 2)
            )


    #
    def close(self):
        self.plane.plot_history()
        if self.display is not None:
            pygame.display.quit()
            pygame.quit()




if __name__ == '__main__':
    print(os.listdir('../Windfields'))
