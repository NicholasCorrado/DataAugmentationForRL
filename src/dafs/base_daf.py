from typing import Dict, List, Any

import numpy as np
from gymnasium.vector import SyncVectorEnv


class BaseDAF:
    """
    Base data augmentation function (DAF) class.
    """
    def __init__(self, env=None, **kwargs):
        self.env = env

        if isinstance(self.env, SyncVectorEnv):
            self.observation_space = env.single_observation_space
        else:
            self.observation_space = env.observation_space

    def _deepcopy_transition(
            self,
            obs: np.ndarray,
            next_obs: np.ndarray,
            action: np.ndarray,
            reward: np.ndarray,
            terminated: np.ndarray,
            infos: List[Dict[str, Any]],
            aug_ratio: int,
    ):
        """
        Deepcopy the input trajectory segment `aug_ratio` times.

        :param obs:
        :param next_obs:
        :param action:
        :param reward:
        :param terminated:
        :param infos:
        :param aug_ratio:
        :return:
        """

        aug_obs = np.tile(obs, (aug_ratio, 1))
        aug_next_obs = np.tile(next_obs, (aug_ratio, 1))
        aug_action = np.tile(action, (aug_ratio, 1))
        aug_reward = np.tile(reward, (aug_ratio,))
        aug_terminated = np.tile(terminated, (aug_ratio,)).astype(bool)
        aug_infos = np.tile([infos], (aug_ratio, 1))

        return aug_obs, aug_next_obs, aug_action, aug_reward, aug_terminated, aug_infos

    def _is_valid(
            self,
            obs: np.ndarray,
            next_obs: np.ndarray,
            action: np.ndarray,
            reward: np.ndarray,
            terminated: np.ndarray,
            infos: List[Dict[str, Any]],
    ):
        """
        Checks if the input transition is a valid DAF input. If there are no restrictions on what transitions can be
        augmented, then let function always return True. Otherwise, override this method and perform necessary checks.

        :param obs:
        :param next_obs:
        :param action:
        :param reward:
        :param terminated:
        :param truncated:
        :param infos:
        :return:
        """
        return True

    def augment(
            self,
            obs: np.ndarray,
            next_obs: np.ndarray,
            action: np.ndarray,
            reward: np.ndarray,
            terminated: np.ndarray,
            infos: List[Dict[str, Any]],
            aug_ratio: int,
            **kwargs,
    ):
        """
        Augment an input trajectory segment.

        :param obs:
        :param next_obs:
        :param action:
        :param reward:
        :param terminated:
        :param infos:
        :param kwargs:
        :return:
        """

        # If the input is not a valid for the given DAF, return None.
        if not self._is_valid(obs, next_obs, action, reward, terminated, infos):
            return None, None, None, None, None, None

        aug_transition = \
            self._deepcopy_transition(obs, next_obs, action, reward, terminated, infos, aug_ratio=aug_ratio)

        self._augment(*aug_transition, aug_ratio=aug_ratio, **kwargs)
        aug_obs, aug_next_obs, aug_action, aug_reward, aug_terminated, aug_infos = aug_transition

        return aug_obs, aug_next_obs, aug_action, aug_reward, aug_terminated, aug_infos

    def _augment(
            self,
            obs: np.ndarray,
            next_obs: np.ndarray,
            action: np.ndarray,
            reward: np.ndarray,
            terminated: np.ndarray,
            infos: List[Dict[str, Any]],
            aug_ratio: int,
            **kwargs,
    ):
        """
        This function augments the input trajectory *in place*.

        :param obs:
        :param next_obs:
        :param action:
        :param reward:
        :param terminated:
        :param infos:
        :param kwargs:
        :return:
        """
        raise NotImplementedError("DAF not implemented.")


