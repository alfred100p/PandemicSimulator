# Confidential, Copyright 2020, Sony Corporation of America, All rights reserved.
from collections import UserString
from typing import List, Optional, Dict, Tuple, Mapping, Type, Sequence

import gym

from .done import DoneFunction
from .interfaces import LocationID, PandemicObservation, NonEssentialBusinessLocationState, PandemicRegulation, \
    InfectionSummary
from .pandemic_sim import PandemicSim
from .reward import RewardFunction, SumReward, RewardFunctionFactory, RewardFunctionType
from .simulator_config import PandemicSimConfig
from .simulator_opts import PandemicSimOpts

__all__ = ['PandemicGymEnv']


class PandemicGymEnv(gym.Env):
    """A gym environment interface wrapper for the Pandemic Simulator."""

    _pandemic_sim: PandemicSim
    _stage_to_regulation: Mapping[int, PandemicRegulation]
    _obs_history_size: int
    _sim_steps_per_regulation: int
    _non_essential_fbusiness_loc_ids: Optional[List[LocationID]]
    _reward_fn: Optional[RewardFunction]
    _done_fn: Optional[DoneFunction]

    _last_observation: PandemicObservation
    _last_reward: float

    def __init__(self,
                 pandemic_sim: PandemicSim,
                 pandemic_regulations: Sequence[PandemicRegulation],
                 reward_fn: Optional[RewardFunction] = None,
                 done_fn: Optional[DoneFunction] = None,
                 obs_history_size: int = 1,
                 sim_steps_per_regulation: int = 1,
                 non_essential_business_location_ids: Optional[List[LocationID]] = None,
                 ):
        """
        :param pandemic_sim: Pandemic simulator instance
        :param pandemic_regulations: A sequence of pandemic regulations
        :param reward_fn: reward function
        :param done_fn: done function
        :param obs_history_size: number of latest sim step states to include in the observation
        :param sim_steps_per_regulation: number of sim_steps to run for each regulation
        :param non_essential_business_location_ids: an ordered list of non-essential business location ids
        """
        self.env=self
        self._max_episode_steps=24
        self._pandemic_sim = pandemic_sim
        self._stage_to_regulation = {reg.stage: reg for reg in pandemic_regulations}
        self._obs_history_size = obs_history_size
        self._sim_steps_per_regulation = sim_steps_per_regulation

        if non_essential_business_location_ids is not None:
            for loc_id in non_essential_business_location_ids:
                assert isinstance(self._pandemic_sim.state.id_to_location_state[loc_id],
                                  NonEssentialBusinessLocationState)
        self._non_essential_business_loc_ids = non_essential_business_location_ids

        self._reward_fn = reward_fn
        self._done_fn = done_fn

        self.action_space = gym.spaces.Discrete(len(self._stage_to_regulation))

    @classmethod
    def from_config(cls: Type['PandemicGymEnv'],
                    sim_config: PandemicSimConfig,
                    pandemic_regulations: Sequence[PandemicRegulation],
                    sim_opts: PandemicSimOpts = PandemicSimOpts(),
                    reward_fn: Optional[RewardFunction] = None,
                    done_fn: Optional[DoneFunction] = None,
                    name: Optional[UserString]=None,
                    obs_history_size: int = 1,
                    non_essential_business_location_ids: Optional[List[LocationID]] = None,
                    ) -> 'PandemicGymEnv':
        """
        Creates an instance using config

        :param sim_config: Simulator config
        :param pandemic_regulations: A sequence of pandemic regulations
        :param sim_opts: Simulator opts
        :param reward_fn: reward function
        :param done_fn: done function
        :param obs_history_size: number of latest sim step states to include in the observation
        :param non_essential_business_location_ids: an ordered list of non-essential business location ids
        """
        sim = PandemicSim.from_config(sim_config, sim_opts, name)

        if sim_config.max_hospital_capacity == -1:
            raise Exception("Nothing much to optimise if max hospital capacity is -1.")

        reward_fn = reward_fn or SumReward(
            reward_fns=[
                RewardFunctionFactory.default(RewardFunctionType.INFECTION_SUMMARY_ABOVE_THRESHOLD,
                                              summary_type=InfectionSummary.CRITICAL,
                                              threshold=sim_config.max_hospital_capacity),
                
                RewardFunctionFactory.default(RewardFunctionType.LOWER_STAGE,
                                              num_stages=len(pandemic_regulations)),
                                             
                RewardFunctionFactory.default(RewardFunctionType.SMOOTH_STAGE_CHANGES,
                                              num_stages=len(pandemic_regulations))
            ],
            weights=[.4, .1,0.02]
            #1, .1, 0.02]
        )

        return PandemicGymEnv(pandemic_sim=sim,
                              pandemic_regulations=pandemic_regulations,
                              sim_steps_per_regulation=sim_opts.sim_steps_per_regulation,
                              reward_fn=reward_fn,
                              done_fn=done_fn,
                              obs_history_size=obs_history_size,
                              non_essential_business_location_ids=non_essential_business_location_ids)

    @property
    def pandemic_sim(self) -> PandemicSim:
        return self._pandemic_sim

    @property
    def observation(self) -> PandemicObservation:
        return self._last_observation

    @property
    def last_reward(self) -> float:
        return self._last_reward

    def step(self, action: int) -> Tuple[PandemicObservation, float, bool, Dict]:
        assert self.action_space.contains(action), "%r (%s) invalid" % (action, type(action))

        # execute the action if different from the current stage
        if action != self._last_observation.stage[-1, 0, 0]:  # stage has a TNC layout
            regulation = self._stage_to_regulation[action]
            self._pandemic_sim.impose_regulation(regulation=regulation)

        # update the sim until next regulation interval trigger and construct obs from state hist
        obs = PandemicObservation.create_empty(
            history_size=self._obs_history_size,
            num_non_essential_business=len(self._non_essential_business_loc_ids)
            if self._non_essential_business_loc_ids is not None else None)

        hist_index = 0
        for i in range(self._sim_steps_per_regulation):
            # step sim
            self._pandemic_sim.step()

            # store only the last self._history_size state values
            if i >= (self._sim_steps_per_regulation - self._obs_history_size):
                obs.update_obs_with_sim_state(self._pandemic_sim.state, hist_index,
                                              self._non_essential_business_loc_ids)
                hist_index += 1

        prev_obs = self._last_observation
        self._last_reward = self._reward_fn.calculate_reward(prev_obs, action, obs) if self._reward_fn else 0.
        done = self._done_fn.calculate_done(obs, action) if self._done_fn else False
        self._last_observation = obs

        #print(self.env._last_observation)
        return self._last_observation, self._last_reward, done, {}

    def reset(self) -> PandemicObservation:
        self._pandemic_sim.reset()
        self._last_observation = PandemicObservation.create_empty(
            history_size=self._obs_history_size,
            num_non_essential_business=len(self._non_essential_business_loc_ids)
            if self._non_essential_business_loc_ids is not None else None)
        self._last_reward = 0.0
        if self._done_fn is not None:
            self._done_fn.reset()
        return self._last_observation.obs

    def render(self, mode: str = 'human') -> bool:
        pass

class PandemicGymEnvWrapper(gym.ActionWrapper):

    _pandemic_sim: PandemicSim
    _stage_to_regulation: Mapping[int, PandemicRegulation]
    _obs_history_size: int
    _sim_steps_per_regulation: int
    _non_essential_business_loc_ids: Optional[List[LocationID]]
    _reward_fn: Optional[RewardFunction]
    _done_fn: Optional[DoneFunction]

    _last_observation: PandemicObservation
    _last_reward: float
    
    def __init__(self,env,warmup):
        super(PandemicGymEnvWrapper,self).__init__(env)
        self.env=env
        self.action_space=gym.spaces.Discrete(3,)
        self.warmup=warmup

    def reset(self) -> PandemicObservation:
        self.env.reset()
        self.warmup=True
        return self.env._last_observation.obs


    def action(self, act):
        '''
        act: action value from [-1,0,1]
            -1 Reduce Stage by 1
            0 Maintain Current Stage
            1 Increase Stage by 1 
        '''
        if self.warmup:
            #print('warmup')
            if self.env._last_observation.obs[11]:
                if self.env._last_observation.stage[0,0,0]<4:
                    return 1+int(self.env._last_observation.stage[0,0,0])
                else:
                    self.warmup=False
                    #print('warmup over'+str(self.env._last_observation.obs[12]*120))
                    
                    return int(self.env._last_observation.stage[0,0,0])
            else:
                if self.env._last_observation.stage[0,0,0]>0:
                    return int(self.env._last_observation.stage[0,0,0])-1
                else:
                    return int(self.env._last_observation.stage[0,0,0])
                
        else:
            return int(max(0,min(act-1+self.env._last_observation.stage[-1, 0, 0],len(self.env._stage_to_regulation)-1)))

    def raction(self, act):
        '''
        act: action value from [-1,0,1]
            -1 Reduce Stage by 1
            0 Maintain Current Stage
            1 Increase Stage by 1 
        '''

        return int(max(0,min(act-1+self.env._last_observation.stage[-1, 0, 0],len(self.env._stage_to_regulation)-1)))



    def step(self, act: int) -> Tuple[PandemicObservation, float, bool, Dict]:
        assert self.action_space.contains(act), "%r (%s) invalid" % (act, type(act))
        act=self.action(act)
        

        # execute the action if different from the current stage
        if act != self.env._last_observation.stage[-1, 0, 0]:  # stage has a TNC layout
            regulation = self.env._stage_to_regulation[act]
            self.env._pandemic_sim.impose_regulation(regulation=regulation)

        # update the sim until next regulation interval trigger and construct obs from state hist
        obs = PandemicObservation.create_empty(
            history_size=self.env._obs_history_size,
            num_non_essential_business=len(self.env._non_essential_business_loc_ids)
            if self.env._non_essential_business_loc_ids is not None else None)

        hist_index = 0
        for i in range(self.env._sim_steps_per_regulation):
            # step sim
            self.env._pandemic_sim.step()

            # store only the last self._history_size state values
            if i >= (self.env._sim_steps_per_regulation - self.env._obs_history_size):
                obs.update_obs_with_sim_state(self.env._pandemic_sim.state, hist_index,
                                              self.env._non_essential_business_loc_ids)
                hist_index += 1

        prev_obs = self.env._last_observation
        self.env._last_reward = self.env._reward_fn.calculate_reward(prev_obs, act, obs) if self.env._reward_fn else 0.
        done = self.env._done_fn.calculate_done(obs, act) if self.env._done_fn else False
        self.env._last_observation = obs
        #print(self.env._last_observation)
        #self.env._pandemic_sim.viz.record(self.env._last_observation,self.env._last_reward)
        #print(self.env._last_reward)
        return self.env._last_observation.obs, self.env._last_reward, done, {}
    
    def rstep(self, act: int) -> Tuple[PandemicObservation, float, bool, Dict]:
        assert self.action_space.contains(act), "%r (%s) invalid" % (act, type(act))
        act=self.raction(act)
        

        # execute the action if different from the current stage
        if act != self.env._last_observation.stage[-1, 0, 0]:  # stage has a TNC layout
            regulation = self.env._stage_to_regulation[act]
            self.env._pandemic_sim.impose_regulation(regulation=regulation)

        # update the sim until next regulation interval trigger and construct obs from state hist
        obs = PandemicObservation.create_empty(
            history_size=self.env._obs_history_size,
            num_non_essential_business=len(self.env._non_essential_business_loc_ids)
            if self.env._non_essential_business_loc_ids is not None else None)

        hist_index = 0
        for i in range(self.env._sim_steps_per_regulation):
            # step sim
            self.env._pandemic_sim.step()

            # store only the last self._history_size state values
            if i >= (self.env._sim_steps_per_regulation - self.env._obs_history_size):
                obs.update_obs_with_sim_state(self.env._pandemic_sim.state, hist_index,
                                              self.env._non_essential_business_loc_ids)
                hist_index += 1

        prev_obs = self.env._last_observation
        self.env._last_reward = self.env._reward_fn.calculate_reward(prev_obs, act, obs) if self.env._reward_fn else 0.
        done = self.env._done_fn.calculate_done(obs, act) if self.env._done_fn else False
        self.env._last_observation = obs
        #print(self.env._last_observation)
        #self.env._pandemic_sim.viz.record(self.env._last_observation,self.env._last_reward)
        #print(self.env._last_reward)

        return self.env._last_observation.obs, self.env._last_reward, done, {}
    
    def plot(self):
        return 0
        #self.env._pandemic_sim.viz.plot()