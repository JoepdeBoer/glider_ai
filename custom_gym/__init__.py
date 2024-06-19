from gym.envs.registration import register
register(
   id='CompEnv-v0',
   entry_point='custom_gym.env:CompEnv',
)