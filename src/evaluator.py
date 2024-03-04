import os

import numpy as np
import torch

class Evaluator:
    def __init__(
            self,
            actor,
            eval_env,
            save_dir,
            n_eval_episodes: int = 20,
            deterministic: bool = True,
            save_model: bool = False,
            device=None,
    ):

        self.actor = actor
        self.eval_env = eval_env

        self.save_dir = save_dir
        self.save_path = f"{self.save_dir}/evaluations.npz"
        self.n_eval_episodes = n_eval_episodes
        self.deterministic = deterministic
        self.save_model = save_model
        self.device = device

        # Data will be saved to ``evaluations.npz``
        os.makedirs(name=self.save_dir, exist_ok=True)

        self.eval_returns = []
        self.eval_timesteps = []
        self.eval_successes = []

    def evaluate(self, timestep):
        returns, successes = self._evaluate()

        self.eval_timesteps.append(timestep)
        self.eval_returns.append(returns)
        self.eval_successes.append(successes)

        np.savez(
            self.save_path,
            timesteps=self.eval_timesteps,
            returns=self.eval_returns,
            successes=self.eval_successes,
        )

        mean_reward, std_reward = np.mean(returns), np.std(returns)
        mean_success, std_success = np.mean(successes), np.std(successes)

        print(f"Timestep={timestep}")
        print(f"Return = {mean_reward:.2f} +/- {std_reward:.2f}")
        print(f"Success Rate = {mean_success:.2f} +/- {std_success:.2f}")
        print()

        return mean_reward, std_reward

    def _evaluate(self):
        eval_returns = []
        eval_successes = []

        obs, _ = self.eval_env.reset()
        for episode_i in range(self.n_eval_episodes):
            ep_rewards = []
            ep_successes = []
            done = False
            step = 0
            while not done:
                step += 1
                # ALGO LOGIC: put action logic here
                with torch.no_grad():
                    actions = self.actor(torch.Tensor(obs).to(self.device))
                    actions = actions.cpu().numpy().clip(self.eval_env.action_space.low,
                                                         self.eval_env.action_space.high)

                next_obs, rewards, terminateds, truncateds, infos = self.eval_env.step(actions)
                done = terminateds or truncateds

                obs = next_obs

                ep_rewards.append(rewards.item())
                ep_successes.append(infos.get('is_success', False)
)

            eval_returns.append(np.sum(ep_rewards))
            eval_successes.append(np.sum(ep_successes))

        return eval_returns, eval_successes

