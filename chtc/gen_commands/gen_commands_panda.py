import os
'''
Recreating Panda envs in ICLR Fig.3 
    1xPolicy:  buffer_size 1e6
    2xViaTranslateGoal: buffer_size 2e6
    2xPolicy: buffer_size 2e6, 2x batch, 2x total_timesteps 
'''

# 7,10,python3*ddpg.py*--env_id*PandaPickAndPlace-v3*--batch_size*512*--save_subdir*1xPolicy*--buffer_size*1000000*
# 7,10,python3*ddpg.py*--env_id*PandaPickAndPlace-v3*--batch_size*512*--save_subdir*2xViaTranslateGoal*--daf*RelabelGoal*--buffer_size*2000000*

# python3 ddpg.py --env_id PandaSlide-v3 --daf RelabelGoal --save_subdir 2xTranslateGoal 

def gen_command(env_id, daf, batch_size, subdir, timesteps):
    python_command = f'python3 ddpg.py --env_id {env_id} ' \
                    f'--batch_size {batch_size} ' \
                    f'--save_subdir {subdir} '
    if daf != None:
        python_command = python_command + f'--daf {daf} ' 
        python_command = python_command + f'--buffer_size {int(2e6)} ' # x2 buffer
    else: # no DAF
        python_command = python_command + f'--buffer_size {int(1e6)} ' # buffer=1M

    if timesteps != None:
        python_command = python_command + f'--total_timesteps {int(timesteps)}'

    mem = 9
    disk = 10
    command = f"{mem},{disk},{python_command}"
    return command

def gen_command_more(env_id, daf, batch_size, subdir, buffer_size, train_freq, timesteps):
    python_command = f'python3 ddpg.py --env_id {env_id} ' \
                    f'--batch_size {batch_size} ' \
                    f'--train_freq {train_freq} ' \
                    f'--save_subdir {subdir} ' \
                    f'--buffer_size {int(buffer_size)} '

    if timesteps != None:
        python_command = python_command + f'--total_timesteps {int(timesteps)}' 
    if daf != None:
        python_command = python_command + f'--daf {daf} ' 

    mem = 9
    disk = 10
    command = f"{mem},{disk},{python_command}"
    return command

if __name__ == "__main__":
    os.makedirs('../commands', exist_ok=True)
    f = open(f"../commands/train_panda.txt", "w")

    env_ids = ['PandaPush-v3', 'PandaSlide-v3']
    daf = 'RelabelGoal'
    for env in env_ids:
        command = gen_command(env, daf=None, batch_size=256, subdir='1xPolicy', timesteps=None) # no DA
        print(command)
        f.write(command.replace(' ', '*') + "\n")
        
        command = gen_command(env, daf=daf, batch_size=256*2, subdir='2xViaTranslateGoal', timesteps=None) # 
        print(command)
        f.write(command.replace(' ', '*') + "\n")

        # 2xPolicy -> 2x buffer, batch, and total_timesteps
        command = gen_command_more(env, daf=None, batch_size=256*2, subdir='2xPolicy', \
                                        buffer_size=2e6, train_freq=4, timesteps=2e6) 
        print(command)
        f.write(command.replace(' ', '*') + "\n")

    # env_ids = ['PandaPickAndPlace-v3', 'PandaFlip-v3']
    env_ids = ['PandaPickAndPlace-v3']
    for env in env_ids:
        command = gen_command(env, daf=None, batch_size=512, subdir='1xPolicy', timesteps=1.5e6) 
        print(command)
        f.write(command.replace(' ', '*') + "\n")
        
        command = gen_command(env, daf, batch_size=512*2, subdir='2xViaTranslateGoal', timesteps=1.5e6) 
        print(command)
        f.write(command.replace(' ', '*') + "\n")

        # 2xPolicy -> 2x buffer, batch, and total_timesteps
        command = gen_command_more(env, daf=None, batch_size=512*2, subdir='2xPolicy', \
                                    buffer_size=2e6, train_freq=4, timesteps=3e6) 
        print(command)
        f.write(command.replace(' ', '*') + "\n")
