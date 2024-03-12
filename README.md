# WeakDataAugmentation

```commandline
cd WeakDataAugmentation
pip install -e .
cd src
pip install -e custom-envs
pip install panda-gym
... # pip install package dependencies
```

## Training Example

```commandline
python ddpg.py --env_id Hopper-v4 --total_timesteps 1000000 --eval_freq 10000 
```
