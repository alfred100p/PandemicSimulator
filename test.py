import os
import sys
from os.path import dirname, abspath
sys.path.append(dirname(dirname(abspath(__file__))))

import gym

from agents.actor_critic_agents.A2C import A2C
from agents.DQN_agents.Dueling_DDQN import Dueling_DDQN
from agents.actor_critic_agents.SAC_Discrete import SAC_Discrete
from agents.actor_critic_agents.A3C import A3C
from agents.policy_gradient_agents.PPO import PPO
from agents.Trainer import Trainer
from utilities.data_structures.Config import Config
from agents.DQN_agents.DDQN import DDQN
from agents.DQN_agents.DDQN_With_Prioritised_Experience_Replay import DDQN_With_Prioritised_Experience_Replay
from agents.DQN_agents.DQN import DQN
from agents.DQN_agents.DQN_With_Fixed_Q_Targets import DQN_With_Fixed_Q_Targets
from tqdm import trange

import pandemic_simulator as ps

ps.init_globals(seed=0)
sim_config = ps.sh.small_town_config

viz = ps.viz.GymViz.from_config(sim_config=sim_config)

env = ps.env.PandemicGymEnv.from_config(name='test', sim_config=sim_config, pandemic_regulations=ps.sh.austin_regulations,done_fn=ps.env.done.ORDone(done_fns=[ps.env.done.InfectionSummaryAboveThresholdDone(summary_type=ps.env.infection_model.InfectionSummary.CRITICAL,threshold=sim_config.max_hospital_capacity*3),ps.env.done.NoPandemicDone(num_days=30)]))


p_env=  ps.env.PandemicGymEnvWrapper(env=env,warmup=True)

config = Config()
config.seed = 1
config.environment = p_env
config.num_episodes_to_run = 8000
config.file_to_save_data_results = "results/data_and_graphs/PandemicSim_Results_Data.pkl"
config.file_to_save_results_graph = "results/data_and_graphs/PandemicSim_Results_Graph.png"
config.show_solution_score = False
config.visualise_individual_results = False
config.visualise_overall_agent_results = True
config.standard_deviation_results = 1.0
config.runs_per_agent = 1
config.use_GPU = True
config.overwrite_existing_results_file = False
config.randomise_random_seed = True
config.save_model = True


config.hyperparameters = {
    
    "Actor_Critic_Agents":  {
        "discount_rate": 0.99,
        "epsilon_decay_rate_denominator": 1.0,
        "normalise_rewards": True,
        "exploration_worker_difference": 2.0,
        "clip_rewards": False,

        "Actor": {

            "learning_rate": 0.01,

            "linear_hidden_units": [128],
            "final_layer_activation": "Softmax",
            "batch_norm": False,
            "tau": 0.005,
            "gradient_clipping_norm": 5,
            "initialiser": "Xavier"
        },

        "Critic": {
            "learning_rate": 0.03,

            "linear_hidden_units": [128],
            "final_layer_activation": None,
            "batch_norm": False,
            "buffer_size": 1000,
            "tau": 0.005,
            "gradient_clipping_norm": 5,
            "initialiser": "Xavier"
        },

        "min_steps_before_learning": 1000,
        "batch_size": 16,
        "mu": 0.0, #for O-H noise
        "theta": 0.15, #for O-H noise
        "sigma": 0.25, #for O-H noise
        "update_every_n_steps": 1,
        "learning_updates_per_learning_session": config.num_episodes_to_run,
        "automatically_tune_entropy_hyperparameter": True,
        "entropy_term_weight": 0.01,
        "add_extra_noise": False,
        "do_evaluation_iterations": True
    }
}
    

if __name__ == "__main__":

    #os.mkdir(os.path.dirname(os.getcwd())+'/plots')

    AGENTS = [SAC_Discrete, ]#DDQN, Dueling_DDQN, DQN, DQN_With_Fixed_Q_Targets,
              #DDQN_With_Prioritised_Experience_Replay, A2C, PPO, A3C ]
    trainer = Trainer(config, AGENTS)
    trainer.run_games_for_agents(load=False)
