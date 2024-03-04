import numpy as np


def check_valid(env, obs, next_obs, action, reward, terminated, truncated, info):
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
    env.reset()

    # Set the environment state to match the input observation. Below is a template for how you would do this for a
    # MuJoCo task.
    qpos, qvel = None, None
    env.set_state(qpos, qvel)

    # determine ture next_obs, reward
    true_next_obs, true_reward, true_terminated, true_truncated, true_info = env.step(action)
    
    if not np.allclose(next_obs, true_next_obs):
        print('Dynamics do not match:', next_obs-true_next_obs)
    if not np.allclose(reward, true_reward):
        print('Rewards do not match:', reward-true_reward)
    if not np.allclose(terminated, true_terminated):
        print('Termination signals do not match:', terminated, true_terminated)
