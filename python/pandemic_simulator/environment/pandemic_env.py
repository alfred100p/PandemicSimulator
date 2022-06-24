# Confidential, Copyright 2020, Sony Corporation of America, All rights reserved.
from typing import List, Optional, Dict, Tuple, Mapping, Type, Sequence

import gym
from istype import isinstanceof
import numpy as np

from .done import DoneFunction, ORDone, InfectionSummaryAboveThresholdDone, NoPandemicDone

from .interfaces import LocationID, PandemicObservation, NonEssentialBusinessLocationState, PandemicRegulation, InfectionSummary, sorted_infection_summary
from .pandemic_sim import PandemicSim
from .reward import RewardFunction, SumReward, RewardFunctionFactory, RewardFunctionType
from .simulator_config import PandemicSimConfig
from .simulator_opts import PandemicSimOpts
from collections import UserString

__all__ = ['PandemicGymEnv', 'PandemicGymEnv3Act']


class PandemicGymEnv(gym.Env):
    """A gym environment interface wrapper for the Pandemic Simulator."""

    _pandemic_sim: PandemicSim
    _stage_to_regulation: Mapping[int, PandemicRegulation]
    _sim_steps_per_regulation: int
    _non_essential_business_loc_ids: Optional[List[LocationID]]
    _reward_fn: Optional[RewardFunction]
    _done_fn: Optional[DoneFunction]
    _critical_flag: bool
    _infected_flag: bool
    _show_day: bool
    _show_gis: bool
    _random_init_population: bool
    _seed: int

    _last_observation: np.array
    _last_state: np.array
    _last_reward: float

    _obs_size: int
    _state_size: int

    obs_index_dict: dict
    state_index_dict: dict

    action_space: gym.spaces.Space
    observation_space: gym.spaces.Space

    def __init__(self,
                 pandemic_sim: PandemicSim,
                 pandemic_regulations: Sequence[PandemicRegulation],
                 reward_fn: Optional[RewardFunction] = None,
                 done_fn: Optional[DoneFunction] = None,
                 sim_steps_per_regulation: int = 24,
                 critic_true_state = False,
                 show_gis = False,
                 show_day = True,
                 flags: list = ["critical"],
                 initial_stage = 0,
                 random_init_population = False,
                 seed = 0,
                 ):
        """
        :param pandemic_sim: Pandemic simulator instance
        :param pandemic_regulations: A sequence of pandemic regulations
        :param reward_fn: reward function
        :param done_fn: done function
        :param obs_history_size: number of latest sim step states to include in the observation
        :param sim_steps_per_regulation: number of sim_steps to run for each regulation. Here each sim step corresponds 
        to an hour and as actions are taken every day it is set to 24
        :param critic_true_state: boolean for whether the critic (if present) views the true state of simulation instead of observation
        :param show_gis: boolean for whether the gis is included in observation
        :param show_day: boolean for whether the current day (integer) is included in observation
        :param flags: list containing strings of flags used in observation
        """
        self._pandemic_sim = pandemic_sim
        self._stage_to_regulation = {reg.stage: reg for reg in pandemic_regulations}
        self._sim_steps_per_regulation = sim_steps_per_regulation
        self._random_init_population = random_init_population

        self._reward_fn = reward_fn
        self._done_fn = done_fn
        self._critic_true_state = critic_true_state
        self._seed = 0

        self.action_space = gym.spaces.Discrete(len(self._stage_to_regulation))

        obs_index = 0
        state_index = 0
        self.obs_index_dict = dict()
        self.state_index_dict = dict()

        self.obs_index_dict['stage'] = obs_index
        self.state_index_dict['stage'] = state_index
        obs_index += 1
        state_index += 1
        
        self.obs_index_dict['gts'] = obs_index
        self.state_index_dict['gts'] = state_index
        obs_index += len(sorted_infection_summary)
        state_index += len(sorted_infection_summary)

        if show_gis:
            self.obs_index_dict['gis'] = obs_index
            obs_index += len(sorted_infection_summary)

        self._show_gis = show_gis
        self.state_index_dict['gis'] = state_index
        state_index += len(sorted_infection_summary)

        if "critical" in flags:
            self._critical_flag = True
            self.obs_index_dict['critical_flag'] = obs_index
            obs_index += 1
        else:
            self._critical_flag = False

        self.state_index_dict['critical_flag'] = state_index
        state_index += 1
        
        if "infected" in flags:
            self._critical_flag = True
            self.obs_index_dict['infected_flag'] = obs_index
            obs_index += 1
        else:
            self._infected_flag = False

        self.state_index_dict['infected_flag'] = state_index
        state_index += 1
        
        self.show_day = show_day
        if show_day:
            self.obs_index_dict['day'] = obs_index
            obs_index += 1
        
        self.state_index_dict['day'] = state_index
        state_index += 1

        self._obs_size = obs_index
        self._state_size = state_index

        self.observation_space = gym.spaces.Discrete(obs_index)        
        self.state_space = gym.spaces.Discrete(state_index)

        state = np.zeros(self._state_size)
        state[self.state_index_dict['stage']] = initial_stage
        state[self.state_index_dict['gis']:self.state_index_dict['gis'] + len(sorted_infection_summary)] = list(self._pandemic_sim.state.global_infection_summary.values())
        state[self.state_index_dict['gts']:self.state_index_dict['gts'] + len(sorted_infection_summary)] = list(self._pandemic_sim.state.global_testing_state.summary.values())
        state[self.state_index_dict['critical_flag']] = self._pandemic_sim.state.critical_above_threshold
        state[self.state_index_dict['infected_flag']] = self._pandemic_sim.state.infected_above_threshold
        state[self.state_index_dict['day']] = self._pandemic_sim.state.sim_time.day
        self._last_state = state
        self._last_observation = PandemicGymEnv.state2obs(self, state)
    
        
        

    @classmethod
    def from_config(cls: Type['PandemicGymEnv'],
                    sim_config: PandemicSimConfig,
                    pandemic_regulations: Sequence[PandemicRegulation],
                    sim_opts: PandemicSimOpts = PandemicSimOpts(),
                    reward_fn: Optional[RewardFunction] = None,
                    done_fn: Optional[DoneFunction] = None,
                    name: Optional[UserString]=None,
                    critic_true_state = False,
                    show_gis = False,
                    show_day = True,
                    flags: list = ["critical"],
                    initial_stage = 0,
                    random_init_population = False,
                    seed = 0) -> 'PandemicGymEnv':
        """
        Creates an instance using config

        :param sim_config: Simulator config
        :param pandemic_regulations: A sequence of pandemic regulations
        :param sim_opts: Simulator opts
        :param reward_fn: reward function
        :param done_fn: done function
        :param obs_history_size: number of latest sim step states to include in the observation
        :param non_essential_business_location_ids: an ordered list of non-essential business location ids
        :param critic_true_state: if an actor critic metod is being used and if the critic views the true environment state not just the observation, this refers to if critic can view global infection summary not just gloabl testing summary.
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
            weights=[.4, .1, 0.02]
        )

        if done_fn==None:
            done_fn=ORDone(done_fns=[
                     InfectionSummaryAboveThresholdDone(summary_type=InfectionSummary.CRITICAL,threshold=sim_config.max_hospital_capacity*3),
                     NoPandemicDone(num_days=30)
                     ])

        return PandemicGymEnv(pandemic_sim=sim,
                              pandemic_regulations=pandemic_regulations,
                              sim_steps_per_regulation=sim_opts.sim_steps_per_regulation,
                              reward_fn=reward_fn,
                              done_fn=done_fn,
                              critic_true_state=critic_true_state,
                              show_gis=show_gis,
                              show_day=show_day,
                            flags=flags,
                            initial_stage=initial_stage,
                            random_init_population = random_init_population,
                            seed = seed,)

    @property
    def pandemic_sim(self) -> PandemicSim:
        return self._pandemic_sim

    @property
    def observation(self) -> np.array:
        return self._last_observation

    def print_obs(self):

        print("Current Stage: "+str(self._last_observation[self.obs_index_dict['stage']]))
        print("Not Infected: "+str(self._last_observation[self.obs_index_dict['gts']+3]))
        print("Infected: "+str(self._last_observation[self.obs_index_dict['gts']+2]))
        print("Critical: "+str(self._last_observation[self.obs_index_dict['gts']]))
        print("Recovered: "+str(self._last_observation[self.obs_index_dict['gts']+4]))
        print("Dead: "+str(self._last_observation[self.obs_index_dict['gts']+1]))
        if self._critical_flag:
            print("Critical Flag: "+str(bool(self._last_observation[self.obs_index_dict['critical_flag']])))
        if self._infected_flag:
            print("Infected Flag: "+str(bool(self._last_observation[self.obs_index_dict['infected_flag']])))
        if self.show_day:
            print("Day: "+str(self._last_observation[self.obs_index_dict['day']]))
        
        return None
    
    def print_state(self):

        print("Current Stage: "+str(self._last_state[self.obs_index_dict['stage']]))
        print("Actual Not Infected: "+str(self._last_state[self.obs_index_dict['gis']+3]))
        print("Actual Infected: "+str(self._last_state[self.obs_index_dict['gis']+2]))
        print("Actual Critical: "+str(self._last_state[self.obs_index_dict['gis']]))
        print("Actual Recovered: "+str(self._last_state[self.obs_index_dict['gis']+4]))
        print("Actual Dead: "+str(self._last_state[self.obs_index_dict['gis']+1]))
        print("Tested Not Infected: "+str(self._last_state[self.obs_index_dict['gts']]))
        print("Tested Infected: "+str(self._last_state[self.obs_index_dict['gts']+1]))
        print("Tested Critical: "+str(self._last_state[self.obs_index_dict['gts']+2]))
        print("Tested Recovered: "+str(self._last_state[self.obs_index_dict['gts']+3]))
        print("Tested Dead: "+str(self._last_state[self.obs_index_dict['gts']+4]))
        print("Critical Flag: "+str(self._last_state[self.obs_index_dict['critical_flag']]))
        print("Infected Flag: "+str(self._last_state[self.obs_index_dict['infected_flag']]))
        print("Day: "+str(self._last_state[self.obs_index_dict['day']]))
        
        return None
        
        
        
        
    
    @property
    def state(self) -> np.array:
        return self._last_state

    @property
    def last_reward(self) -> float:
        return self._last_reward

    def state2obs(self, state: np.array) -> np.array:
        obs = np.zeros(self._state_size)
        obs[self.state_index_dict['stage']] = state[self.state_index_dict['stage']]
        obs[self.state_index_dict['gts']:self.state_index_dict['gts']+len(sorted_infection_summary)] = state[self.state_index_dict['gts']:self.state_index_dict['gts']+len(sorted_infection_summary)]
        if self._show_gis:
            obs[self.state_index_dict['gis']:self.state_index_dict['gis']+len(sorted_infection_summary)] = state[self.state_index_dict['gis']:self.state_index_dict['gis']+len(sorted_infection_summary)]
        if self. show_day:
            obs[self.state_index_dict['day']] = state[self.state_index_dict['day']]
        if self._critical_flag:
            obs[self.state_index_dict['critical_flag']] = state[self.state_index_dict['critical_flag']]
        if self._infected_flag:
            obs[self.state_index_dict['infected_flag']] = state[self.state_index_dict['infected_flag']]
        
        return obs

    def step(self, action: int) -> Tuple[PandemicObservation, float, bool, Dict]:
        assert self.action_space.contains(action), "%r (%s) invalid" % (action, type(action))

        # execute the action if different from the current stage
        if action != self._last_observation[self.obs_index_dict['stage']]:  # stage has a TNC layout
            regulation = self._stage_to_regulation[action]
            self._pandemic_sim.impose_regulation(regulation=regulation)

        # update the sim until next regulation interval trigger and construct obs from state hist

        for i in range(self._sim_steps_per_regulation):
            # step sim
            self._pandemic_sim.step()

        state = np.zeros(self._state_size)
        state[self.state_index_dict['stage']] = action
        state[self.state_index_dict['gis']:self.state_index_dict['gis'] + len(sorted_infection_summary)] = list(self._pandemic_sim.state.global_infection_summary.values())
        state[self.state_index_dict['gts']:self.state_index_dict['gts'] + len(sorted_infection_summary)] = list(self._pandemic_sim.state.global_testing_state.summary.values())
        state[self.state_index_dict['critical_flag']] = self._pandemic_sim.state.critical_above_threshold
        state[self.state_index_dict['infected_flag']] = self._pandemic_sim.state.infected_above_threshold
        state[self.state_index_dict['day']] = self._pandemic_sim.state.sim_time.day
        

        prev_state = self._last_state
        self._last_reward = self._reward_fn.calculate_reward(prev_state, action, state, self.state_index_dict) if self._reward_fn else 0.
        done = self._done_fn.calculate_done(state, action, self.state_index_dict) if self._done_fn else False
        self._last_state = state
        self._last_observation = PandemicGymEnv.state2obs(self, state)

        return self._last_observation, self._last_reward, done, self._last_state, {}

    def reset(self) -> PandemicObservation:
        self._seed += 1
        self._pandemic_sim.reset(random = self._random_init_population, seed = self._seed)
        self._last_observation = np.zeros(self._obs_size)
        '''PandemicObservation.create_empty(
            history_size=self._obs_history_size,
            num_non_essential_business=len(self._non_essential_business_loc_ids)
            if self._non_essential_business_loc_ids is not None else None)'''
        self._last_reward = 0.0
        if self._done_fn is not None:
            self._done_fn.reset()
        return self._last_observation

    def render(self, mode: str = 'human') -> bool:
        pass

class PandemicGymEnv3Act(gym.ActionWrapper):
    def __init__(self, env: PandemicGymEnv):
        super().__init__(env)
        self.env = env

        self.action_space = gym.spaces.Discrete(3, start=-1)

    @classmethod
    def from_config(self,
                    name: UserString,
                    sim_config: PandemicSimConfig,
                    pandemic_regulations: Sequence[PandemicRegulation],
                    sim_opts: PandemicSimOpts = PandemicSimOpts(),
                    reward_fn: Optional[RewardFunction] = None,
                    done_fn: Optional[DoneFunction] = None,
                    critic_true_state = False,
                    show_gis = False,
                    show_day = True,
                    flags: list = ["critical"],
                    initial_stage = 0,
                    ) -> 'PandemicGymEnv3Act':
        self.action_space = gym.spaces.Discrete(3, start=-1)
        self.env = PandemicGymEnv.from_config(sim_config = sim_config,
        name = name,
        pandemic_regulations=pandemic_regulations,
        sim_opts = sim_opts,
        reward_fn=reward_fn,
        done_fn=done_fn,
        critic_true_state=critic_true_state,
        show_gis=show_gis,
        show_day=show_day,
        flags=flags,
        initial_stage=initial_stage)

        return self
    
    def step(self, action):
        return self.env.step(int(self.action(self, action)))

    def action(self, action):
        assert self.action_space.contains(action), "%r (%s) invalid" % (action, type(action))
        return min(4, max(0, self.env._last_state[self.env.state_index_dict['stage']] + action))