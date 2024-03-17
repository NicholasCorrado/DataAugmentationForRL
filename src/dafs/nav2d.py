import numpy as np
import gymnasium as gym
import custom_envs

from typing import Dict, List, Any

from src.dafs.base_daf import BaseDAF
from src.dafs.validate import check_valid


class TranslateAgent(BaseDAF):
    def __init__(self, env=None, **kwargs):
        super().__init__(env, **kwargs)

    def _augment(
            self,
            # aug_ratio: int,
            obs: np.ndarray,
            next_obs: np.ndarray,
            action: np.ndarray,
            reward: np.ndarray,
            terminated: np.ndarray,
            truncated: np.ndarray,
            infos: List[Dict[str, Any]],
            **kwargs, ):
        # sample rand pos
        new_pos_x = np.random.uniform(low=-1, high=1)
        new_pos_y = np.random.uniform(low=-1, high=1)

        # compute new state
        delta_x = next_obs[:, 0] - obs[:, 0]
        delta_y = next_obs[:, 1] - obs[:, 1]
        obs[:, 0] = new_pos_x
        obs[:, 1] = new_pos_y

        # compute next_state
        next_obs[:, 0] = new_pos_x + delta_x
        next_obs[:, 1] = new_pos_y + delta_y

        # vector of distances from goal
        dist_from_goal = np.linalg.norm(next_obs[:, 0:2] - next_obs[:, 2:4])

        # vector of bool at_goal true/false
        terminated[:] = dist_from_goal < 0.05

        # masking where reward=1 if terminated, -0.1 else
        reward[terminated] = +1.0
        reward[~terminated] = -0.1

        # validat

if __name__ == "__main__":
    env_id = 'Nav2d-v0'
    env = gym.make(env_id)

    aug_func = TranslateAgent(env=env)

    num_steps = int(1e4)

    for t in range(num_steps):
        obs, info = env.reset()

        action = env.action_space.sample()
        next_obs, reward, terminated, truncated, info = env.step(action)
        aug_obs, aug_next_obs, aug_action, aug_reward, aug_terminated, aug_truncated, aug_infos = aug_func.augment(
            obs, next_obs, action, reward, terminated, truncated, info)

        check_valid(env,
                    obs=aug_obs[0],
                    action=aug_action[0],
                    reward=aug_reward[0],
                    next_obs=aug_next_obs[0],
                    terminated=aug_terminated[0],
                    truncated=aug_truncated[0],
                    info=aug_infos[0])
        