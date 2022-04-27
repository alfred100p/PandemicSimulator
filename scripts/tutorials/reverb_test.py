import reverb
import tensorflow as tf
import pandemic_simulator as ps
from tqdm import trange




env = ps.env.PandemicGymEnv.from_config(sim_config, pandemic_regulations=ps.sh.austin_regulations,done_fn=ps.env.done.ORDone(done_fns=[ps.env.done.InfectionSummaryAboveThresholdDone(summary_type=ps.env.infection_model.InfectionSummary.CRITICAL,threshold=sim_config.max_hospital_capacity*3),ps.env.done.NoPandemicDone(num_days=30)]))
viz = ps.viz.GymViz.from_config(sim_config=sim_config)

num_iterations = 1000

initial_collect_steps = 1000
collect_steps_per_iteration = 1
replay_buffer_capacity = 1000

batch_size = 32

critic_learining_rate = 1e-5
actor_learning_rate = 1e-5
alpha_learning_rate = 1e-5
targe_update_tau = 0.05
target_update_period = 1
gamma = 0.99
reward_scale_factor = 1.0

actor_fc_layer_params = (256)
critic_joint_fc_layer_params = (256)

log_interval = 5000

num_eval_episodes = 30
eval_interval = 20

policy_save_interval = 50

'''
1000 sample 10 episode
training step 1000
'''


env.reset()
for _ in trange(120, desc='Simulating day'):
    obs, reward, done, aux = env.step(action=random.randint(0,4))  # here the action is the discrete regulation stage identifier
    viz.record((obs, reward))
    if done:
        break