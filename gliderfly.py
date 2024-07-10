import gymnasium as gym
from stable_baselines3 import PPO
from tkinter import filedialog
from custom_gym.Agent import Agent
from custom_gym.environement import Windfield


models_dir = "models/"

plane = Agent(100., 100., 1500., 30., 0., -0.72, 28.825, -1.55, 43.24, -3.1, 52.84, 18.33, 69.44, (0.,0.))
Wind = Windfield()
Wind.load_field()
env = gym.make('CompEnv-v0', wind=Wind, agent=plane, time_limit=1000, render_mode='training')

model_path = filedialog.askopenfilename(initialdir = models_dir)
model = PPO.load(model_path, env=env)

episodes = 5

for ep in range(episodes):
    obs = env.reset()[0]
    done = False
    while not done:
        action, _states = model.predict(obs)
        obs, rewards, done, _, info = env.step(action)
        env.render()