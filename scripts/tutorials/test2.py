# Confidential, Copyright 2020, Sony Corporation of America, All rights reserved.

from tqdm import trange

import pandemic_simulator as ps
import random


def run_pandemic_gym_env() -> None:
    """Here we execute the gym envrionment wrapped simulator using austin regulations,
    a small town config and default person routines."""

    print('\nA tutorial that runs the OpenAI Gym environment wrapped simulator', flush=True)

    
    total_reward = 0
    # select a simulator config
    sim_config = ps.sh.small_town_config

    episodes = 10

    
    l=[]
    for j in range(episodes):
        # init globals
        ps.init_globals(seed=j)

        

        # make env
            # make env
        env = ps.env.PandemicGymEnv.from_config(sim_config, pandemic_regulations=ps.sh.austin_regulations)

        # run stage-0 action steps in the environment
        env.reset()
        for i in trange(100, desc='Simulating day'):
            obs, reward, done, aux = env.step(action=0)  # here the action is the discrete regulation stage identifier
            
            if obs.critical_above_threshold[...,0]:
                l+=[i]
                break
    ########################################################################################################################################
        # generate plots

    for el in l:
        print(el)
        


if __name__ == '__main__':
    run_pandemic_gym_env()

