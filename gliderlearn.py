import os
import time
from datetime import datetime

from stable_baselines3 import PPO
import gymnasium as gym

from custom_gym.Agent import Agent
from custom_gym.environement import Windfield


windfield = Windfield() # in reset() windfield is loaded
agent = Agent(0, 0, 1500, 30, 0, -0.72, 28.825, -1.55, 43.24, -3.1, 52.84, 18.33, 69.44, (-10000, 10000), time_step= 0.1)


models_dir = f"models/{int(time.time())}/"
logdir = f"logs/{int(time.time())}/"

if not os.path.exists(models_dir):
	os.makedirs(models_dir)

if not os.path.exists(logdir):
	os.makedirs(logdir)

env = gym.make('CompEnv-v0', wind=Windfield(), agent=Agent(100, 100, 1500, 30, 0, -0.72, 28.825, -1.55, 43.24, -3.1, 52.84, 18.33, 69.44, (-10000, 10000), time_step= 0.1), time_limit=10000, render_mode='human')
env.reset()

model = PPO('MlpPolicy', env, verbose=1, tensorboard_log=logdir)

TIMESTEPS = 10000
iters = 0
while True:
	iters += 1
	model.learn(total_timesteps=TIMESTEPS, reset_num_timesteps=False, tb_log_name=f"PPO")
	model.save(f"{models_dir}/{TIMESTEPS*iters}")