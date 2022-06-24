# Confidential, Copyright 2020, Sony Corporation of America, All rights reserved.

from tqdm import trange

import pandemic_simulator as ps
import random


def run_pandemic_gym_env() -> None:
    """Here we execute the gym envrionment wrapped simulator using austin regulations,
    a small town config and default person routines."""

    print('\nA tutorial that runs the OpenAI Gym environment wrapped simulator', flush=True)

    # init globals
    ps.init_globals(seed=104923490)

    # select a simulator config
    sim_config = ps.sh.small_town_config

    # make env

    wrap = ps.env.PandemicGymEnv3Act.from_config(sim_config = sim_config, pandemic_regulations=ps.sh.austin_regulations, name='t1')

    # setup viz
    viz2 = ps.viz.GymViz.from_config(sim_config=sim_config)
    sim_viz2 = ps.viz.SimViz.from_config(sim_config=sim_config)

    # run stage-0 action steps in the environment
    wrap.env.reset()
    for i in trange(120, desc='Simulating day'):
        wrap.env.print_obs()
        if i>9 and i%10==0:
            viz2.plot()
            sim_viz2.plot()
        if i==0:
            action = 0 
        else:
#######################################################################################################################################            
            #Replace the code in this if else statement with your own policy based on observation
            '''
            Access Critical flag: obs[wrap.env.observation[wrap.env.obs_index_dict['critical_flag']]]
            Access Infected flag: obs[wrap.env.observation[wrap.env.obs_index_dict['infected_flag']]]
            Access Stage: obs[wrap.env.observation[wrap.env.obs_index_dict['stage']]]
            Access Day: obs[wrap.env.observation[wrap.env.obs_index_dict['day']]]
            Access Not Infected Population(based on testing): obs[wrap.env.observation[wrap.env.obs_index_dict['gts']+3]]
            Access Infected Population(based on testing): obs[wrap.env.observation[wrap.env.obs_index_dict['gts']+2]]
            Access Critical Population(based on testing): obs[wrap.env.observation[wrap.env.obs_index_dict['gts']]]
            Access Dead Population(based on testing): obs[wrap.env.observation[wrap.env.obs_index_dict['gts']+1]]
            Access Recovered Population(based on testing): obs[wrap.env.observation[wrap.env.obs_index_dict['gts']+4]]
            '''
            if obs[wrap.env.observation[wrap.env.obs_index_dict['critical_flag']]]:
                action = 1
            elif obs[wrap.env.observation[wrap.env.obs_index_dict['gts']+2]]<20:
                action = 0
            else:
                action = -1
#######################################################################################################################################
        obs, reward, done, state, aux = wrap.step(wrap, action=int(action))  # here the action is the discrete regulation stage identifier
        viz2.record((state, reward), wrap.env.state_index_dict)
        sim_viz2.record_state(state = wrap.env.pandemic_sim.state, obs=state, index_dict=wrap.env.state_index_dict)

    # generate plots
    viz2.plot()
    sim_viz2.plot()
    print('Reward:'+str(reward))


if __name__ == '__main__':
    run_pandemic_gym_env()

