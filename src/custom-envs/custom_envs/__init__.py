import os
from gym.envs.registration import register

# ENVS_DIR = os.path.join(os.path.dirname(__file__), 'envs')

register(
    id="Nav2d-v0",
    entry_point="custom_envs.nav2d:Nav2dEnv",
    max_episode_steps=100,
)