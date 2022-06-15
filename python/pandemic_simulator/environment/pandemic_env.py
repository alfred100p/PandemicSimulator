# Confidential, Copyright 2020, Sony Corporation of America, All rights reserved.
from typing import List, Optional, Dict, Tuple, Mapping, Type, Sequence

import gym
import numpy as np

from .done import DoneFunction, ORDone, InfectionSummaryAboveThresholdDone, NoPandemicDone
from .interfaces import LocationID, PandemicObservation, NonEssentialBusinessLocationState, PandemicRegulation, InfectionSummary, sorted_infection_summary
from .pandemic_sim import PandemicSim
from .reward import RewardFunction, SumReward, RewardFunctionFactory, RewardFunctionType
from .simulator_config import PandemicSimConfig
from .simulator_opts import PandemicSimOpts

__all__ = ['PandemicGymEnv']


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

    _last_observation: np.array
    _last_true_state: np.array
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
                 flags: list=["critical"],
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

        self._reward_fn = reward_fn
        self._done_fn = done_fn
        self._critic_true_state = critic_true_state

        self.action_space = gym.spaces.Discrete(len(self._stage_to_regulation))

        obs_index = 0
        state_index = 0

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
        #make sure day is 6th last 1-5 is gis. 6 is day used in done functions
        if show_day:
            self.obs_index_dict['day'] = obs_index
            obs_index += 1
        
        self.state_index_dict['day'] = state_index
        state_index += 1

        self._obs_size = obs_index
        self._state_size = state_index

        self.observation_space = gym.spaces.Discrete(obs_index)        
        self.true_state_space = gym.spaces.Discrete(state_index)
        
        

    @classmethod
    def from_config(cls: Type['PandemicGymEnv'],
                    sim_config: PandemicSimConfig,
                    pandemic_regulations: Sequence[PandemicRegulation],
                    sim_opts: PandemicSimOpts = PandemicSimOpts(),
                    reward_fn: Optional[RewardFunction] = None,
                    done_fn: Optional[DoneFunction] = None,
                    critic_true_state = False,
                    show_gis = False,
                    show_day = True,
                    flags: list=["critical"],
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
        :param critic_true_state: if an actor critic metod is being used and if the critic views the true environment state not just the observation, this refers to if critic can view global infection summary not just gloabl testing summary.
        """
        sim = PandemicSim.from_config(sim_config, sim_opts)

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
                            flags=flags)

    @property
    def pandemic_sim(self) -> PandemicSim:
        return self._pandemic_sim

    @property
    def observation(self) -> np.array:
        return self._last_observation

    @property
    def last_reward(self) -> float:
        return self._last_reward

    def step(self, action: int) -> Tuple[PandemicObservation, float, bool, Dict]:
        assert self.action_space.contains(action), "%r (%s) invalid" % (action, type(action))

        # execute the action if different from the current stage
        if action != self._last_observation[self._stage_index]:  # stage has a TNC layout
            regulation = self._stage_to_regulation[action]
            self._pandemic_sim.impose_regulation(regulation=regulation)

        # update the sim until next regulation interval trigger and construct obs from state hist
        state = np.array()#self._last_true_state
        #obs = self._last_observation
        '''PandemicObservation.create_empty(
            history_size=self._obs_history_size,
            num_non_essential_business=len(self._non_essential_business_loc_ids)
            if self._non_essential_business_loc_ids is not None else None)'''

        hist_index = 0
        for i in range(self._sim_steps_per_regulation):
            # step sim
            self._pandemic_sim.step()

            # store only the last self._history_size state values
            if self._obs_history:
                if i >= (self._sim_steps_per_regulation - self._obs_history_size):
                    obs.update_obs_with_sim_state(self._pandemic_sim.state, hist_index,
                                                self._non_essential_business_loc_ids)
                    hist_index += 1

        prev_state = self._last_true_state
        self._last_reward = self._reward_fn.calculate_reward(prev_state, action, state) if self._reward_fn else 0.
        done = self._done_fn.calculate_done(state, action) if self._done_fn else False
        self._last_observation = state

        return self._last_observation, self._last_reward, done, {}

    def reset(self) -> PandemicObservation:
        self._pandemic_sim.reset()
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
        """Wraps an environment to allow a modular transformation of the :meth:`step` and :meth:`reset` methods.
        Args:
            env: The environment to wrap
        """
        super().__init__(env)
        self.env = env

        self._action_space: Optional[gym.spaces.Space] = None
    
    def step(self, action):
        """Runs the environment :meth:`env.step` using the modified ``action`` from :meth:`self.action`."""
        return self.env.step(self.action(action))

    def action(self, action):
        """Returns a modified action before :meth:`env.step` is called."""
        raise NotImplementedError