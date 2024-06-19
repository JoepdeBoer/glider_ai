import gym
import time
import pygame
from datetime import datetime
from custom_gym.Agent import Agent
from custom_gym.environement import Windfield

# Environement initiaisation parameters
size = 100000
resolution = 10
thermalheight_avg = 1500
thermalheight_std = 50
thermalstrenght_avg = 10
thermalstrenght_std = 0.5
thermalrad_avg = 200
thermalrad_std = 30
name = f'Wind_{datetime.today().strftime("%m-%d-%H-%M")}'

plane = Agent(100, 100, 1500, 30, 0, -0.72, 28.825, -1.55, 43.24, -3.1, 52.84, 18.33, 69.44, "Discus")
Wind = Windfield(size, resolution, thermalheight_avg, thermalheight_std, thermalstrenght_avg, thermalstrenght_std, thermalrad_avg, thermalrad_std)

if __name__ == '__main__':
    env = gym.make('CompEnv-v0', wind=Wind, agent=plane, time_limit=1000, render_mode='human')
    env.reset()
    for _ in range(1000):
        observation, reward, done, _, _ = env.step(env.action_space.sample())
        print('Observation : ' + str(observation))
        print('Reward      : ' + str(reward))
        print('Done        : ' + str(done))
        print('---------------------')

    env.close()