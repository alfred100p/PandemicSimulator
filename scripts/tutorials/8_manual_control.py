# Confidential, Copyright 2020, Sony Corporation of America, All rights reserved.

from tqdm import trange

import pandemic_simulator as ps
import random


def run_pandemic_gym_env() -> None:
    """Here we execute the gym envrionment wrapped simulator using austin regulations,
    a small town config and default person routines."""

    print('\nA tutorial that runs the OpenAI Gym environment wrapped simulator', flush=True)

    # init globals
    ps.init_globals(seed=2)

    # select a simulator config
    sim_config = ps.sh.small_town_config

    # make env

    env = ps.env.PandemicGymEnv.from_config(sim_config = sim_config, pandemic_regulations=ps.sh.austin_regulations, name='t1')

    # setup viz
    viz = ps.viz.GymViz.from_config(sim_config=sim_config)
    sim_viz = ps.viz.SimViz.from_config(sim_config=sim_config)

    # run stage-0 action steps in the environment
    env.reset()
    print('''Welcome to Tutorial 1.
    You will manually change the stage of response to the simulated pandemic.''')
    for i in trange(120, desc='Simulating day'):
        env.print_obs()
        if i>9 and i%10==0:
            viz.plot()
            sim_viz.plot()
        action = input('Enter a Stage number from 0-4. 0 is no restrictions, 4 is most strict lockdown.\n')
        obs, reward, done, state, aux = env.step(action=int(action))  # here the action is the discrete regulation stage identifier
        viz.record((state, reward), env.state_index_dict)
        sim_viz.record_state(state = env.pandemic_sim.state, obs=state, index_dict=env.state_index_dict)

    # generate plots
    viz.plot()
    sim_viz.plot()
    print('Reward:'+str(reward))


if __name__ == '__main__':
    run_pandemic_gym_env()

