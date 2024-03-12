from typing import Dict, List, Any

import numpy as np
import gymnasium as gym
import panda_gym
from gymnasium.vector import SyncVectorEnv
from gymnasium.wrappers import FlattenObservation

import src.dafs.panda_push
from src.dafs.base_daf import BaseDAF

class RelabelGoalBase(BaseDAF):
    """
    Base data augmentation function (DAF) class.
    """
    def __init__(self, env=None, **kwargs):
        super().__init__(env, **kwargs)

        if isinstance(env, SyncVectorEnv):
            observation_space = env.single_observation_space
        else:
            observation_space = env.observation_space

        self.desired_goal_mask = np.zeros(observation_space.shape).astype(bool)
        self.achieved_goal_mask = np.zeros(observation_space.shape).astype(bool)

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
            **kwargs,
    ):
        """
        This function augments the input trajectory *in place*.

        :param obs:
        :param next_obs:
        :param action:
        :param reward:
        :param terminated:
        :param truncated:
        :param infos:
        :param kwargs:
        :return:
        """

        # sample new goal from the task's goal space.
        new_goal = np.array([self.env.task._sample_goal()])

        # relabel goal in obs and next_obs
        obs[:, self.desired_goal_mask] = new_goal
        next_obs[:, self.desired_goal_mask] = new_goal

        # compute reward now that the goal has changed
        achieved_goal = next_obs[:, self.achieved_goal_mask]
        reward[:] = self.env.task.compute_reward(achieved_goal, new_goal, infos)

        # terminated = true only when the agent successfully solves the task.
        is_success = self.env.task.is_success(achieved_goal, new_goal).astype(bool)
        terminated[:] = is_success

        # action and truncated remain unchanged.
