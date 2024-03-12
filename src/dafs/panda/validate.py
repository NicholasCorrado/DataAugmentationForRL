import numpy as np
import gymnasium as gym
from gymnasium.wrappers import FlattenObservation

# import src.dafs.panda.push
from src.dafs.panda import push, slide, pickandplace, flip


def check_valid(env, obs, action, next_obs, reward, terminated, truncated, info):
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

    # determine ture next_obs, reward
    true_next_obs, true_reward, true_terminated, true_truncated, true_info = env.step(action)

    if not np.allclose(next_obs, true_next_obs):
        print('Dynamics do not match:', next_obs - true_next_obs)
    if not np.allclose(reward, true_reward):
        print('Rewards do not match:', reward, true_reward)
    if not np.allclose(terminated, true_terminated):
        print('Termination signals do not match:', terminated, true_terminated)




if __name__ == "__main__":


    env_id = 'PandaFlip-v3'
    env = gym.make(env_id)
    env = FlattenObservation(env)

    aug_func = {
        'PandaPush-v3': push.RelabelGoal,
        'PandaSlide-v3': slide.RelabelGoal,
        'PandaPickAndPlace-v3': pickandplace.RelabelGoal,
        'PandaFlip-v3': flip.RelabelGoal,
    }

    f = aug_func[env_id](env)

    num_steps = int(1e4)

    for t in range(num_steps):
        obs, info = env.reset()
        state_id = env.save_state()

        action = env.action_space.sample()
        next_obs, reward, terminated, truncated, info = env.step(action)
        aug_obs, aug_next_obs, aug_action, aug_reward, aug_terminated, aug_truncated, aug_infos = f.augment(
            obs, next_obs, action, reward, terminated, truncated, info)

        env.reset()
        env.restore_state(state_id)
        env.task.goal = aug_obs[0, f.desired_goal_mask]

        check_valid(env,
                    obs=aug_obs[0],
                    action=aug_action[0],
                    reward=aug_reward[0],
                    next_obs=aug_next_obs[0],
                    terminated=aug_terminated[0],
                    truncated=aug_truncated[0],
                    info=aug_infos[0])