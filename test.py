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
from python.pandemic_simulator.environment.pandemic_env import PandemicGymEnvWrapper
from tqdm import trange

import pandemic_simulator as ps

ps.init_globals(seed=0)
sim_config = ps.sh.small_town_config

viz = ps.viz.GymViz.from_config(sim_config=sim_config)

env = ps.env.PandemicGymEnv.from_config(name='test', sim_config=sim_config, pandemic_regulations=ps.sh.austin_regulations,done_fn=ps.env.done.ORDone(done_fns=[ps.env.done.InfectionSummaryAboveThresholdDone(summary_type=ps.env.infection_model.InfectionSummary.CRITICAL,threshold=sim_config.max_hospital_capacity*3),ps.env.done.NoPandemicDone(num_days=30),ps.env.done.NoMoreInfectionsDone()]))


p_env=  PandemicGymEnvWrapper(env=env,warmup=True)

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
config.use_GPU = False
config.overwrite_existing_results_file = False
config.randomise_random_seed = True
config.save_model = True


config.hyperparameters = {
    
    
    "Stochastic_Policy_Search_Agents": {
        "policy_network_type": "Linear",
        "noise_scale_start": 1e-2,
        "noise_scale_min": 1e-3,
        "noise_scale_max": 2.0,
        "noise_scale_growth_factor": 2.0,
        "stochastic_action_decision": False,
        "num_policies": 1,
        "episodes_per_policy": 1,
        "num_policies_to_keep": 5,
        "clip_rewards": False
    },
    "Policy_Gradient_Agents": {
        "learning_rate": 0.05,
        "linear_hidden_units": [20, 20],
        "final_layer_activation": "SOFTMAX",
        "learning_iterations_per_round": 5,
        "discount_rate": 0.99,
        "batch_norm": False,
        "clip_epsilon": 0.1,
        "episodes_per_learning_round": 4,
        "normalise_rewards": True,
        "gradient_clipping_norm": 7.0,
        "mu": 0.0, #only required for continuous action games
        "theta": 0.0, #only required for continuous action games
        "sigma": 0.0, #only required for continuous action games
        "epsilon_decay_rate_denominator": 1.0,
        "clip_rewards": False
    },


    
    "Actor_Critic_Agents":  {

        "learning_rate": 0.005,
        "linear_hidden_units": [20, 10],
        "final_layer_activation": ["SOFTMAX", None],
        "gradient_clipping_norm": 5.0,
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
            "learning_rate": 0.1,

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
        "discount_rate": 0.99,
        "mu": 0.0, #for O-H noise
        "theta": 0.15, #for O-H noise
        "sigma": 0.25, #for O-H noise
        "action_noise_std": 0.2,  # for TD3
        "action_noise_clipping_range": 0.5,  # for TD3
        "update_every_n_steps": 30,
        "learning_updates_per_learning_session": 1,
        "automatically_tune_entropy_hyperparameter": True,
        "entropy_term_weight": None,
        "add_extra_noise": False,
        "do_evaluation_iterations": True
    }
}
    

if __name__ == "__main__":

    #os.mkdir(os.path.dirname(os.getcwd())+'/plots')

    AGENTS = [SAC_Discrete, ]#DDQN, Dueling_DDQN, DQN, DQN_With_Fixed_Q_Targets,
              #DDQN_With_Prioritised_Experience_Replay, A2C, PPO, A3C ]
    trainer = Trainer(config, AGENTS)
    trainer.run_games_for_agents(load=True)
