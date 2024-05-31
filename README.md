# WeakDataAugmentation

```commandline
cd DataAugmentationForRL
pip install -e .
pip install -e src/custom_envs
pip install gymnasium panda-gym torch tyro pyyaml stable-baselines3 tensorboard gymnasium_robotics rliable
```

## Training Example

```commandline
python ddpg.py --env_id Hopper-v4 --total_timesteps 1000000 --eval_freq 10000 
python sac.py --env_id Hopper-v4 --total_timesteps 1000000 --eval_freq 10000 
python td3.py --env_id Hopper-v4 --total_timesteps 1000000 --eval_freq 10000 
```
