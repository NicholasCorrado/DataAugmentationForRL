import math
from typing import List, Dict, Any, NamedTuple

import numpy as np
import gymnasium as gym
from gymnasium.wrappers import FlattenObservation
from gymnasium_robotics.envs.maze import PointMazeEnv

from src.dafs.base_daf import BaseDAF


class Cell(NamedTuple):
    xlo: int
    xhi: int
    ylo: int
    yhi: int

class TranslateRotate(BaseDAF):
    def __init__(self, env, **kwargs):
        super().__init__(env=env, **kwargs)
        self.effective_cell_radius = 0.39
        self.effective_wall_radius = 0.5 + (0.5 - self.effective_cell_radius)

        self.goal = None
        self.valid_cells = []
        self.wall_cells = []

        self.maze_map = np.array(self.env.maze.maze_map)

        length, width = self.env.maze.map_length, self.env.maze.map_width
        for i in range(length):  # row
            for j in range(width):  # col
                cell_type = self.maze_map[i, j]
                if cell_type == 0:
                    self.valid_cells.append(np.array([i, j]))
                else:
                    self.wall_cells.append(np.array([i, j]))
        self.valid_cells = np.array(self.valid_cells)

        self.cell_bounds = {}
        for (r,c) in self.valid_cells:
            xlo, ylo = -0.5, -0.5
            xhi, yhi = +0.5, +0.5

            if self.maze_map[r+1, c] == 1:
                ylo = -0.39
            if self.maze_map[r-1, c] == 1:
                yhi = +0.39
            if self.maze_map[r, c+1] == 1:
                xhi = +0.39
            if self.maze_map[r, c-1] == 1:
                xlo = +0.39

            self.cell_bounds[(r,c)] = Cell(xlo, xhi, ylo, yhi)

        self.thetas = np.array([0, np.pi/2, np.pi, np.pi*3/2])
        self.rotation_matrices = []
        for theta in self.thetas:
            M = np.array([
                [np.cos(theta), -np.sin(theta)],
                [np.sin(theta), np.cos(theta)]
            ])
            self.rotation_matrices.append(M)

        self.achieved_goal_mask = np.array([0, 1])
        self.desired_goal_mask = np.array([2, 3])
        self.pos_mask = np.array([4, 5])
        self.vel_mask = np.array([6, 7])

    def _sample_rotation_matrix(self, **kwargs):
        idx = np.random.randint(len(self.rotation_matrices))
        return self.rotation_matrices[idx]

    def _is_valid_cell_pos(self, obs):
        pos = obs[:, self.pos_mask]
        cell_rowcol = self._cell_xy_to_rowcol(pos)
        cell_pos = self._cell_rowcol_to_xy(cell_rowcol)

        if np.all(np.abs(pos - cell_pos) <= self.effective_cell_radius):
            return True
        return False

    def _is_valid(
            self,
            obs: np.ndarray,
            next_obs: np.ndarray,
            action: np.ndarray,
            reward: np.ndarray,
            terminated: np.ndarray,
            infos: List[Dict[str, Any]],
    ):

        return True
        # return self._is_valid_cell_pos(obs) and self._is_valid_cell_pos(next_obs)

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

        cell_rowcol = self._sample_valid_cells(aug_ratio)
        new_pos = self._cell_rowcol_to_xy(cell_rowcol)
        new_pos += self._sample_xy_position_noise(cell_rowcol=cell_rowcol, range=self.effective_cell_radius)

        # compute change in xy position (displacement)
        delta_pos = next_obs[:, self.pos_mask] - obs[:, self.pos_mask]

        # rotate the displacement vector
        M = self._sample_rotation_matrix()
        delta_pos = (M @ delta_pos.T).T

        # rotate action
        action[:] = (M @ action.T).T

        # rotate velocity
        obs[:, self.vel_mask] = (M @ obs[:, self.vel_mask].T).T
        next_obs[:, self.vel_mask] = (M @ next_obs[:, self.vel_mask].T).T

        # translate positions
        obs[:, self.pos_mask] = new_pos
        obs[:, self.achieved_goal_mask] = new_pos

        next_obs[:, self.pos_mask] = new_pos + delta_pos
        next_obs[:, self.achieved_goal_mask] = new_pos + delta_pos

        # compute reward
        reward[:] = self.env.compute_reward(next_obs[:, self.achieved_goal_mask], next_obs[:, self.desired_goal_mask], {})

        # compute termination signal
        terminated[:] = self.compute_terminated(next_obs[:, self.achieved_goal_mask], next_obs[:, self.desired_goal_mask], {})

    def _sample_valid_cells(self, n: int):
        idx = np.random.choice(len(self.valid_cells), size=(n,))
        return self.valid_cells[idx]

    def _cell_rowcol_to_xy(self, rowcol_pos: np.ndarray) -> np.ndarray:
        """Converts a cell index `(i,j)` to x and y coordinates in the MuJoCo simulation"""
        x = (rowcol_pos[:, [1]] + 0.5) * self.env.maze.maze_size_scaling - self.env.maze.x_map_center
        y = self.env.maze.y_map_center - (rowcol_pos[:, [0]] + 0.5) * self.env.maze.maze_size_scaling

        return np.hstack([x, y])

    def _cell_xy_to_rowcol(self, xy_pos: np.ndarray) -> np.ndarray:
        """Converts a cell x and y coordinates to `(i,j)`"""
        i = np.floor((self.env.maze.y_map_center - xy_pos[:, [1]]) / self.env.maze.maze_size_scaling)
        j = np.floor((xy_pos[:, [0]] + self.env.maze.x_map_center) / self.env.maze.maze_size_scaling)
        return np.hstack([i, j])

    def _sample_xy_position_noise(self, cell_rowcol: np.ndarray, range: float) -> np.ndarray:
        """Pass an x,y coordinate and it will return the same coordinate with a noise addition
        sampled from a uniform distribution
        """
        xlo, ylo = -range, -range
        xhi, yhi = range, range

        return np.random.uniform(low=[xlo, ylo], high=[xhi, yhi], size=(len(cell_rowcol), 2)) * self.env.maze.maze_size_scaling

    def compute_terminated(
        self, achieved_goal: np.ndarray, desired_goal: np.ndarray, info
    ) -> bool:
        if not self.env.continuing_task:
            # If task is episodic terminate the episode when the goal is reached
            return np.linalg.norm(achieved_goal - desired_goal, axis=-1) <= 0.45
        else:
            # Continuing tasks don't terminate, episode will be truncated when time limit is reached (`max_episode_steps`)
            return False

class RelabelGoal(TranslateRotate):

    def __init__(self, env, **kwargs):
        super().__init__(env, **kwargs)

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
        cell_rowcol = self._sample_valid_cells(aug_ratio)
        new_goal = self._cell_rowcol_to_xy(cell_rowcol)
        new_goal += self._sample_xy_position_noise(cell_rowcol=cell_rowcol, range=0.25)

        # relabel goa1
        obs[:, self.desired_goal_mask] = new_goal
        next_obs[:, self.desired_goal_mask] = new_goal

        # compute reward
        reward[:] = self.env.compute_reward(next_obs[:, self.achieved_goal_mask], next_obs[:, self.desired_goal_mask], {})

        # compute termination signal
        terminated[:] = self.compute_terminated(next_obs[:, self.achieved_goal_mask], next_obs[:, self.desired_goal_mask], {})


class TranslateRotateRelabelGoal(TranslateRotate):

    def __init__(self, env, **kwargs):
        super().__init__(env, **kwargs)

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

        super()._augment(obs, next_obs, action, reward, terminated, infos, aug_ratio)

        cell_rowcol = self._sample_valid_cells(aug_ratio)
        new_goal = self._cell_rowcol_to_xy(cell_rowcol)
        new_goal += self._sample_xy_position_noise(cell_rowcol=cell_rowcol, range=0.25)

        # relabel goal
        obs[:, self.desired_goal_mask] = new_goal
        next_obs[:, self.desired_goal_mask] = new_goal

        # compute reward
        reward[:] = self.env.compute_reward(next_obs[:, self.achieved_goal_mask], next_obs[:, self.desired_goal_mask], {})

        # compute termination signal
        terminated[:] = self.compute_terminated(next_obs[:, self.achieved_goal_mask], next_obs[:, self.desired_goal_mask], {})


def check_valid(env, obs, next_obs, action, reward, terminated, info):
    """
    Check that an input transition is valid:
      1. next state is correct, i.e. s' = p(s,a)
      2. reward is correct, i.e. r = r(s,a)
      3. termination signal is correct, i.e. terminated=True only when the episode is actually terminated

    The truncated signal does not need to be checked, but we include it as an input anyways for completeness.
    The info dict may contain useful state/reward/termination information, so we include it as an input.

    :param env:
    :param obs:
    :param next_obs:
    :param action:
    :param reward:
    :param terminated:
    :param truncated:
    :param info:
    :return:
    """

    # Set the environment state to match the input observation. Below is a template for how you would do this for a
    # MuJoCo task.
    env.point_env.set_state(qpos=obs[4:6], qvel=obs[6:])
    env.unwrapped.goal = obs[2:4]

    # determine ture next_obs, reward
    true_next_obs, true_reward, true_terminated, true_truncated, true_info = env.step(action)

    if not np.allclose(next_obs, true_next_obs):
        print('Dynamics do not match:', next_obs - true_next_obs)
    if not np.allclose(reward, true_reward):
        print('Rewards do not match:', reward - true_reward)
    if not np.allclose(terminated, true_terminated):
        print('Termination signals do not match:', terminated, true_terminated)


if __name__ == "__main__":

    env = gym.make('PointMaze_Medium-v3', render_mode='human')
    env = FlattenObservation(env)

    num_steps = int(1e4)
    aug_func = RelabelGoal(env)

    for t in range(num_steps):
        obs, info = env.reset()

        action = env.action_space.sample()
        next_obs, reward, terminated, truncated, info = env.step(action)

        obs = np.expand_dims(obs, axis=0)
        action = np.expand_dims(action, axis=0)
        reward = np.expand_dims(reward, axis=0)
        next_obs = np.expand_dims(next_obs, axis=0)
        terminated = np.expand_dims(terminated, axis=0)

        aug_obs, aug_next_obs, aug_action, aug_reward, aug_terminated, aug_infos = aug_func.augment(
            obs, next_obs, action, reward, terminated, info, aug_ratio=4)

        if aug_obs is not None:
            check_valid(env,
                        obs=aug_obs[0],
                        action=aug_action[0],
                        reward=aug_reward[0],
                        next_obs=aug_next_obs[0],
                        terminated=aug_terminated[0],
                        info=aug_infos[0])


