# Confidential, Copyright 2020, Sony Corporation of America, All rights reserved.

from tqdm import trange

import pandemic_simulator as ps
import random


def run_pandemic_gym_env() -> None:
    """Here we execute the gym envrionment wrapped simulator using austin regulations,
    a small town config and default person routines."""

    print('\nA tutorial that runs the OpenAI Gym environment wrapped simulator', flush=True)

    # init globals
    ps.init_globals(seed=0)

    # select a simulator config
    sim_config = ps.sh.small_town_config

    '''# make env
    env = ps.env.PandemicGymEnv.from_config(sim_config = sim_config, pandemic_regulations=ps.sh.austin_regulations)

    # setup viz
    viz = ps.viz.GymViz.from_config(sim_config=sim_config)
    sim_viz = ps.viz.SimViz.from_config(sim_config=sim_config)

    # run stage-0 action steps in the environment
    env.reset()
    for _ in trange(10, desc='Simulating day'):
        obs, reward, done, state, aux = env.step(action=random.randint(0,4))  # here the action is the discrete regulation stage identifier
        viz.record((state, reward), env.state_index_dict)
        sim_viz.record_state(state = env.pandemic_sim.state, obs=state, index_dict=env.state_index_dict)

    # generate plots
    viz.plot()
    sim_viz.plot()
    '''

    wrap = ps.env.PandemicGymEnv3Act.from_config(sim_config = sim_config, pandemic_regulations=ps.sh.austin_regulations)

    # setup viz
    viz2 = ps.viz.GymViz.from_config(sim_config=sim_config)
    sim_viz2 = ps.viz.SimViz.from_config(sim_config=sim_config)

    # run stage-0 action steps in the environment
    wrap.env.reset()
    for _ in trange(120, desc='Simulating day'):
        obs, reward, done, state, aux = wrap.step(wrap, action=random.randint(-1,1))  # here the action is the discrete regulation stage identifier
        viz2.record((state, reward), wrap.env.state_index_dict)
        sim_viz2.record_state(state = wrap.env.pandemic_sim.state, obs=state, index_dict=wrap.env.state_index_dict)

    # generate plots
    viz2.plot()
    sim_viz2.plot()


if __name__ == '__main__':
    run_pandemic_gym_env()
