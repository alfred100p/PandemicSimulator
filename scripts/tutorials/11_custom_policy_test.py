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

    # setup viz
    viz = []
    sim_viz = []
    for i in range(episodes):
        viz += [ps.viz.GymViz.from_config(sim_config=sim_config)]
        sim_viz += [ps.viz.SimViz.from_config(sim_config=sim_config)]

    

    for j in range(episodes):
        # init globals
        ps.init_globals(seed=j)

        

        # make env

        wrap = ps.env.PandemicGymEnv3Act.from_config(sim_config = sim_config, pandemic_regulations=ps.sh.austin_regulations)        

        # run stage-0 action steps in the environment
        wrap.env.reset()
        Reward = 0
        for i in trange(120, desc='Simulating day'):
            
            if i==0:
                action = 0 

            else:                
                #######################################################################################################################################            
                #Replace the code in the below if-else statement with your own policy
                if obs.time_day[...,0]>20:
                    action = 1
                elif not obs.infection_above_threshold:
                    action = 0
                else:
                    action = -1
                ########################################################################################################################################
            obs, reward, done, aux = wrap.step(action=int(action))  # here the action is the discrete regulation stage identifier
            print(obs)
            Reward += reward
            viz[j].record((obs, reward))
            sim_viz[j].record_state(state = wrap.env.pandemic_sim.state)
        # generate plots
        
        print('Reward:'+str(Reward))
        total_reward += Reward

    for i in range(episodes):
        viz[i].plot()
        sim_viz[i].plot()
    print('Avg Reward:'+str(total_reward/episodes))


if __name__ == '__main__':
    run_pandemic_gym_env()

