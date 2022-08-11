import pandemic_simulator as ps

ps.init_globals(seed=111111)

sim_config = ps.sh.small_town_config
# make env
wrap = ps.env.PandemicGymEnv3Act.from_config(sim_config = sim_config, pandemic_regulations=ps.sh.austin_regulations)       


# run stage-0 action steps in the environment
init_state = wrap.reset()
import pdb; pdb.set_trace()
