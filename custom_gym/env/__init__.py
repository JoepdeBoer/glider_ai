from gym.envs.registration import register
from custom_gym.env.gymenv import CompEnv

register(
   id='CompEnv-v0',
   entry_point='custom_gym.env:CompEnv',
)