from gymnasium import register

register(
    id="Nav2d-v0",
    entry_point="custom_envs.nav2d:Nav2dEnv",
    max_episode_steps=100,
)