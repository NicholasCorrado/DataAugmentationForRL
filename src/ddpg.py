# docs and experiment results can be found at https://docs.cleanrl.dev/rl-algorithms/ddpg/#ddpg_continuous_actionpy
import os
import random
import time
from dataclasses import dataclass, field
from typing import Optional, Union

import gymnasium as gym
import panda_gym, gymnasium_robotics, custom_envs
from src.dafs import DAFS

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import tyro
import yaml
from stable_baselines3.common.buffers import ReplayBuffer, ReplayBufferSamples
from torch.utils.tensorboard import SummaryWriter

from src.utils import get_latest_run_id
from src.evaluator import Evaluator

@dataclass
class Args:
    # wandb tracking
    exp_name: str = os.path.basename(__file__)[: -len(".py")]  # experiment name
    track: bool = False             # if toggled, experiment will be tracked on wandb
    wandb_project_name: str = "cleanRL" # wandb's project name
    wandb_entity: Optional[str] = None # the entity (team) of wandb's project
    capture_video: bool = False # whether to capture videos of the agent performances (check `videos` folder)

    # experiment config
    torch_deterministic: bool = True # if toggled, `torch.backends.cudnn.deterministic=False`
    cuda: bool = True # if toggled, cuda will be enabled by default
    env_id: str = "PandaPickAndPlace-v3" # environment id of the Atari game

    env_kwargs: dict[str, Union[bool, float, str]] = field(default_factory=dict)
    """
    usage: --env_kwargs arg1 val1 arg2 val2 arg3 val3
    
    To make PointMaze tasks use a sparse reward function:
        --env_kwargs continuing_task False
    """
    # env_kwargs: str = "arg1:one arg2:two" # additional keyword arguments to be passed to the env constructor
    total_timesteps: int = int(1e6)         # total timesteps of the experiments
    eval_freq: int = 10000                  # num of timesteps between policy evals
    n_eval_episodes: int = 50               # num of eval episodes
    seed: Optional[int] = None              # seed of the experiment
    run_id: Optional[int] = None
    save_rootdir: str = "results"                   # top-level directory where results will be saved
    save_subdir: Optional[str] = "2xTranslateGoal"  # lower level directories
    save_dir: str = field(init=False)               # the lower-level directories 
    save_model: bool = False # whether to save model into the `runs/{run_name}` folder

    # Algorithm specific arguments
    learning_rate: float = 3e-4     # learning rate of optimizer
    buffer_size: int = int(2e6)     # replay memory buffer size
    gamma: float = 0.99             # discount factor gamma
    tau: float = 0.005              # target smoothing coefficient (default: 0.005)
    batch_size: int = 512           # batch size of sample from the reply memory
    exploration_noise: float = 0.1  # scale of exploration noise
    # learning_starts: int = 25e3     # timestep to start learning
    learning_starts: int = 0     # timestep to start learning
    policy_frequency: int = 2       # frequency of training policy (delayed)
    noise_clip: float = 0.5         # noise clip parameter of the Target Policy Smoothing Regularization
    random_action_prob: float = 0.0 # probability of sampling a random action

    # DA hyperparams
    daf: Optional[str] = "RelabelGoal"
    alpha: float = 0.50
    aug_ratio: int = 16

    def __post_init__(self):

        self.save_dir = f"{self.save_rootdir}/{self.env_id}/ddpg/{self.save_subdir}"
        if self.run_id is None:
            self.run_id = get_latest_run_id(save_dir=self.save_dir) + 1
        self.save_dir += f"/run_{self.run_id}"

        if self.seed is None:
            self.seed = self.run_id
        else:
            self.seed = np.random.randint(2 ** 32 - 1)

        # dump training config to save dir
        os.makedirs(self.save_dir, exist_ok=True)
        with open(os.path.join(self.save_dir, "config.yml"), "w") as f:
            yaml.dump(self, f, sort_keys=True)


def make_env(env_id, env_kwargs, seed, idx, capture_video, run_name):
    def thunk():
        if capture_video and idx == 0:
            env = gym.make(env_id, render_mode="rgb_array", **env_kwargs)
            # env = Nav2dEnv()
            env = gym.wrappers.RecordVideo(env, f"videos/{run_name}")
        else:
            env = gym.make(env_id, **env_kwargs)
            # env = Nav2dEnv()

        # Flatten Dict obs so we don't need to handle them a special case in DA
        if isinstance(env.observation_space, gym.spaces.Dict):
            env = gym.wrappers.FlattenObservation(env)
        env = gym.wrappers.RecordEpisodeStatistics(env)
        env.action_space.seed(seed)
        return env

    return thunk

# ALGO LOGIC: initialize agent here:
class QNetwork(nn.Module):
    def __init__(self, env):
        super().__init__()
        self.fc1 = nn.Linear(np.array(env.single_observation_space.shape).prod() + np.prod(env.single_action_space.shape), 64)
        self.fc2 = nn.Linear(64, 64)
        self.fc3 = nn.Linear(64, 1)

    def forward(self, x, a):
        x = torch.cat([x, a], 1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


class Actor(nn.Module):
    def __init__(self, env):
        super().__init__()
        self.fc1 = nn.Linear(np.array(env.single_observation_space.shape).prod(), 64)
        self.fc2 = nn.Linear(64, 64)
        self.fc_mu = nn.Linear(64, np.prod(env.single_action_space.shape))
        # action rescaling
        self.register_buffer(
            "action_scale", torch.tensor((env.action_space.high - env.action_space.low) / 2.0, dtype=torch.float32)
        )
        self.register_buffer(
            "action_bias", torch.tensor((env.action_space.high + env.action_space.low) / 2.0, dtype=torch.float32)
        )

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = torch.tanh(self.fc_mu(x))
        return x * self.action_scale + self.action_bias


if __name__ == "__main__":
    import stable_baselines3 as sb3

    if sb3.__version__ < "2.0":
        raise ValueError(
            """Ongoing migration: run the following command to install the new dependencies:
poetry run pip install "stable_baselines3==2.0.0a1"
"""
        )
    args = tyro.cli(Args)
    run_name = f"{args.env_id}__{args.exp_name}__{args.seed}__{int(time.time())}"
    if args.track:
        import wandb

        wandb.init(
            project=args.wandb_project_name,
            entity=args.wandb_entity,
            sync_tensorboard=True,
            config=vars(args),
            name=run_name,
            monitor_gym=True,
            save_code=True,
        )
    writer = SummaryWriter(f"runs/{run_name}")
    writer.add_text(
        "hyperparameters",
        "|param|value|\n|-|-|\n%s" % ("\n".join([f"|{key}|{value}|" for key, value in vars(args).items()])),
    )

    # TRY NOT TO MODIFY: seeding
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = args.torch_deterministic

    # device = torch.device("cuda" if torch.cuda.is_available() and args.cuda else "cpu")
    device = torch.device("cpu")

    # env setup
    envs = gym.vector.SyncVectorEnv([make_env(args.env_id, args.env_kwargs, args.seed, 0, args.capture_video, run_name)])
    assert isinstance(envs.single_action_space, gym.spaces.Box), "only continuous action space is supported"

    actor = Actor(envs).to(device)
    qf1 = QNetwork(envs).to(device)
    qf1_target = QNetwork(envs).to(device)
    target_actor = Actor(envs).to(device)
    target_actor.load_state_dict(actor.state_dict())
    qf1_target.load_state_dict(qf1.state_dict())
    q_optimizer = optim.Adam(list(qf1.parameters()), lr=args.learning_rate)
    actor_optimizer = optim.Adam(list(actor.parameters()), lr=args.learning_rate)

    envs.single_observation_space.dtype = np.float32
    rb = ReplayBuffer(
        args.buffer_size,
        envs.single_observation_space,
        envs.single_action_space,
        device,
        # Since we're using gymnasium, we'll use the `terminated` flag to handle timeout vs termination.
        handle_timeout_termination=False,
    )
    
    # @TODO: Initialize empty replay buffer for augmented data
    if args.daf is not None:
        daf = DAFS[args.env_id][args.daf](env=envs.envs[0])
    else:
        daf = None

    aug_rb = ReplayBuffer(
        args.buffer_size,
        envs.single_observation_space,
        envs.single_action_space,
        device,
        handle_timeout_termination=False,
    )

    eval_env = gym.vector.SyncVectorEnv([make_env(args.env_id, args.env_kwargs, 0, 0, False, run_name)])
    evaluator = Evaluator(actor, eval_env, args.save_dir, n_eval_episodes=args.n_eval_episodes)

    start_time = time.time()

    # TRY NOT TO MODIFY: start the game
    obs, _ = envs.reset(seed=args.seed)
    for global_step in range(args.total_timesteps):
        # ALGO LOGIC: put action logic here
        if global_step < args.learning_starts and np.random.random() < args.random_action_prob:
            actions = np.array([envs.single_action_space.sample() for _ in range(envs.num_envs)])
        else:
            with torch.no_grad():
                actions = actor(torch.Tensor(obs).to(device))
                actions += torch.normal(0, actor.action_scale * args.exploration_noise)
                actions = actions.cpu().numpy().clip(envs.single_action_space.low, envs.single_action_space.high)

        # print("actions ", actions)
        # TRY NOT TO MODIFY: execute the game and log data.
        next_obs, rewards, terminations, truncations, infos = envs.step(actions)

        # TRY NOT TO MODIFY: record rewards for plotting purposes
        if "final_info" in infos:
            for info in infos["final_info"]:
                writer.add_scalar("charts/episodic_return", info["episode"]["r"], global_step)
                writer.add_scalar("charts/episodic_length", info["episode"]["l"], global_step)
                break

        # TRY NOT TO MODIFY: save data to reply buffer; handle `final_observation`
        real_next_obs = next_obs.copy()
        for idx, trunc in enumerate(truncations):
            if trunc:
                real_next_obs[idx] = infos["final_observation"][idx]
        rb.add(obs, real_next_obs, actions, rewards, terminations, infos)
        
        ###############
        # sample m augmented samples from a given DAF and append it to the augmented replay buffer
        if daf is not None:
            aug_obs, aug_next_obs, aug_action, aug_reward, aug_terminated, aug_infos = daf.augment(
                obs, real_next_obs, actions, rewards, terminations, infos, aug_ratio=args.aug_ratio)

            aug_rb.extend(aug_obs, aug_next_obs, aug_action, aug_reward, aug_terminated, aug_infos) # doesn't need truncated?
        ##############

        # TRY NOT TO MODIFY: CRUCIAL step easy to overlook
        obs = next_obs

        # ALGO LOGIC: training.
        if global_step > args.learning_starts:
            ###############
            # For a given alpha \in [0, 1] sample (1-alpha)*batch_size samples from the observed replay buffer and
            # alpha*batch_size samples from the augmented replay buffer.
            if daf is not None:
                obs_data_size = int((1 - args.alpha) * args.batch_size) # must be int, not float
                obs_data = rb.sample(obs_data_size)
                # data += aug_rb.sample(int(args.alpha * args.batch_size))
                aug_data = aug_rb.sample(int(args.alpha * args.batch_size))
                # data += aug_data
                # data = rb.sample(args.batch_size)
                # data.add(aug_rb.observations[0], aug_rb.observations[1], aug_rb.actions[0],\
                #         aug_rb.rewards[0], aug_rb.done: np.ndarray)
                observations = torch.concat((obs_data.observations, aug_data.observations))
                actions = torch.concat((obs_data.actions, aug_data.actions))
                next_observations = torch.concat((obs_data.next_observations, aug_data.next_observations))
                rewards = torch.concat((obs_data.rewards, aug_data.rewards))
                dones = torch.concat((obs_data.dones, aug_data.dones))

                data = ReplayBufferSamples(observations, actions, next_observations, dones, rewards)
            ##############
            else:
                data = rb.sample(args.batch_size)

            with torch.no_grad():
                next_state_actions = target_actor(data.next_observations)
                qf1_next_target = qf1_target(data.next_observations, next_state_actions)
                next_q_value = data.rewards.flatten() + (1 - data.dones.flatten()) * args.gamma * (qf1_next_target).view(-1)

            qf1_a_values = qf1(data.observations, data.actions).view(-1)
            qf1_loss = F.mse_loss(qf1_a_values, next_q_value)

            # optimize the model
            q_optimizer.zero_grad()
            qf1_loss.backward()
            q_optimizer.step()

            if global_step % args.policy_frequency == 0:
                actor_loss = -qf1(data.observations, actor(data.observations)).mean()
                actor_optimizer.zero_grad()
                actor_loss.backward()
                actor_optimizer.step()

                # update the target network
                for param, target_param in zip(actor.parameters(), target_actor.parameters()):
                    target_param.data.copy_(args.tau * param.data + (1 - args.tau) * target_param.data)
                for param, target_param in zip(qf1.parameters(), qf1_target.parameters()):
                    target_param.data.copy_(args.tau * param.data + (1 - args.tau) * target_param.data)

                if global_step % 1000 == 0:
                    writer.add_scalar("losses/qf1_values", qf1_a_values.mean().item(), global_step)
                    writer.add_scalar("losses/qf1_loss", qf1_loss.item(), global_step)
                    writer.add_scalar("losses/actor_loss", actor_loss.item(), global_step)
                    # print("SPS:", int(global_step / (time.time() - start_time)))
                    writer.add_scalar("charts/SPS", int(global_step / (time.time() - start_time)), global_step)

        if global_step % args.eval_freq == 0:
            evaluator.evaluate(global_step)


    if args.save_model:
        model_path = f"{args.save_dir}/model"
        torch.save((actor.state_dict(), qf1.state_dict()), model_path)
        print(f"model saved to {model_path}")


    envs.close()
    writer.close()