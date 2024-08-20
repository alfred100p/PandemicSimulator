"""
This file is used to evaluate the simulation itself. It compares the data generated from the simulation to real world data.
"""

from tqdm import trange

import pandemic_simulator as ps
import numpy as np


def run_pandemic_gym_env(num_evaluations = 1, sim_config = ps.sh.small_town_config, pandemic_regulations=ps.sh.austin_regulations, 
                         action_fn=lambda x: 0,agg_fn=lambda x: np.mean(np.abs(x)),score_fn=lambda x: np.mean(np.abs(x))) -> None:

    seed = 0
    scores = []
    np_array =  np.loadtxt('scripts/simulator_experiments/test_data/austin_nyt.csv', delimiter=',', skiprows=1, usecols=[2,3])

    while(seed < num_evaluations):
        score = 0
        ps.init_globals(seed=seed)

        # make env
        env = ps.env.PandemicGymEnv.from_config(sim_config, pandemic_regulations=pandemic_regulations)

        # setup viz
        viz = ps.viz.GymViz.from_config(sim_config=sim_config)

        # run stage-0 action steps in the environment
        env.reset()
        l = []
        for i in trange(100, desc='Simulating day'):
    
            obs, reward, done, aux = env.step(action=action_fn(env))  # here the action is the discrete regulation stage identifier
            viz.record((obs, reward))
            print(np_array[i])
            l += [np_array[i] - obs.global_infection_summary[0,0,1]]

        # generate plots
        viz.plot()


        score = score_fn(np.array(l))

        seed += 1
        scores += [score]
    scores = np.array(scores)
    print("Aggregate Score:" agg_fn(scores))
    print("Scores:" scores)






if __name__ == '__main__':
    run_pandemic_gym_env()