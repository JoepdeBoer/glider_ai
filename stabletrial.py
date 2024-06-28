import faulthandler
import cv2
from stable_baselines3 import PPO
import gymnasium as gym
from custom_gym.Agent import Agent
from custom_gym.environement import Windfield


faulthandler.enable()
timesteps = 100


plane = Agent(100., 100., 1500., 30., 0., -0.72, 28.825, -1.55, 43.24, -3.1, 52.84, 18.33, 69.44, (0.,0.))
Wind = Windfield()
Wind.load_field()
env = gym.make('CompEnv-v0', wind=Wind, agent=plane, time_limit=10000, render_mode='training')
model = PPO('MultiInputPolicy', env, verbose=1)
model.learn(total_timesteps=timesteps, progress_bar=True)
episodes = 10
vec_env = model.get_env()


for ep in range(episodes):
    obs = vec_env.reset()
    done = False
    while not done:
        action, _ = model.predict(obs)
        obs, rewards, done, _, info = env.step(action)
    vec_env.close()