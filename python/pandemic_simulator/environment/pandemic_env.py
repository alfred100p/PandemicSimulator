# Confidential, Copyright 2020, Sony Corporation of America, All rights reserved.
from typing import List, Optional, Dict, Tuple, Mapping, Type, Sequence, cast

import gym
import numpy as np

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
    _day_limit: int
    _non_essential_business_loc_ids: Optional[List[LocationID]]
    _reward_fn: Optional[RewardFunction]
    _done_fn: Optional[DoneFunction]

    _last_observation: np.array
    _last_reward: float

    def __init__(self,
                 pandemic_sim: PandemicSim,
                 pandemic_regulations: Sequence[PandemicRegulation],
                 reward_fn: Optional[RewardFunction] = None,
                 done_fn: Optional[DoneFunction] = None,
                 obs_history_size: int = 1,
                 sim_steps_per_regulation: int = 24,
                 day_limit = 120,
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
        self._pandemic_sim = pandemic_sim
        self._stage_to_regulation = {reg.stage: reg for reg in pandemic_regulations}
        self._obs_history_size = obs_history_size
        self._sim_steps_per_regulation = sim_steps_per_regulation
        self._day_limit = day_limit

        if non_essential_business_location_ids is not None:
            for loc_id in non_essential_business_location_ids:
                assert isinstance(self._pandemic_sim.state.id_to_location_state[loc_id],
                                  NonEssentialBusinessLocationState)
        self._non_essential_business_loc_ids = non_essential_business_location_ids

        self._reward_fn = reward_fn
        self._done_fn = done_fn

        self.action_space = gym.spaces.Discrete(len(self._stage_to_regulation))
        population=len(self.pandemic_sim._persons)
        self.observation_space_dict = gym.spaces.Dict(
            {
                "global infection Summary": gym.spaces.Tuple((gym.spaces.Discrete(population),gym.spaces.Discrete(population),
            gym.spaces.Discrete(population),gym.spaces.Discrete(population),gym.spaces.Discrete(population),gym.spaces.Discrete(population))),
                "global testing Summary": gym.spaces.Tuple((gym.spaces.Discrete(population),gym.spaces.Discrete(population),
            gym.spaces.Discrete(population),gym.spaces.Discrete(population),gym.spaces.Discrete(population),gym.spaces.Discrete(population))),
                "stage": gym.spaces.Discrete(len(self._stage_to_regulation)),
                "day": gym.spaces.Discrete(self._day_limit),
                "critical above threshold" : gym.spaces.Discrete(2)
            }
        )
        self.observation_space = gym.spaces.Tuple(
            #gis
            gym.spaces.Tuple((gym.spaces.Discrete(population),gym.spaces.Discrete(population),gym.spaces.Discrete(population),gym.spaces.Discrete(population),gym.spaces.Discrete(population),gym.spaces.Discrete(population))),
            #gts
            gym.spaces.Tuple((gym.spaces.Discrete(population),gym.spaces.Discrete(population),gym.spaces.Discrete(population),gym.spaces.Discrete(population),gym.spaces.Discrete(population),gym.spaces.Discrete(population))),
            #stage
            gym.spaces.Discrete(len(self._stage_to_regulation)),
            #day
            gym.spaces.Discrete(self._day_limit),
            #critical above threshold
            gym.spaces.Discrete(2)
        )

    @classmethod
    def from_config(cls: Type['PandemicGymEnv'],
                    sim_config: PandemicSimConfig,
                    pandemic_regulations: Sequence[PandemicRegulation],
                    sim_opts: PandemicSimOpts = PandemicSimOpts(),
                    day_limit = 120,
                    reward_fn: Optional[RewardFunction] = None,
                    done_fn: Optional[DoneFunction] = None,
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
        sim = PandemicSim.from_config(sim_config, sim_opts)

        if sim_config.max_hospital_capacity == -1:
            raise Exception("Nothing much to optimise if max hospital capacity is -1.")

        reward_fn = reward_fn or SumReward(
            reward_fns=[
                RewardFunctionFactory.default(RewardFunctionType.INFECTION_SUMMARY_ABOVE_THRESHOLD,
                                              summary_type=InfectionSummary.CRITICAL,
                                              threshold=sim_config.max_hospital_capacity),
                RewardFunctionFactory.default(RewardFunctionType.INFECTION_SUMMARY_ABOVE_THRESHOLD,
                                              summary_type=InfectionSummary.CRITICAL,
                                              threshold=3 * sim_config.max_hospital_capacity),
                RewardFunctionFactory.default(RewardFunctionType.LOWER_STAGE,
                                              num_stages=len(pandemic_regulations)),
                RewardFunctionFactory.default(RewardFunctionType.SMOOTH_STAGE_CHANGES,
                                              num_stages=len(pandemic_regulations))
            ],
            weights=[.4, 1, .1, 0.02]
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
    def observation(self) -> np.array:
        return self._last_observation

    @property
    def last_reward(self) -> float:
        return self._last_reward

    def step(self, action: int) -> Tuple[PandemicObservation, float, bool, Dict]:
        assert self.action_space.contains(action), "%r (%s) invalid" % (action, type(action))

        # execute the action if different from the current stage
        if action != self._last_observation[10]:  # stage has a TNC layout
            regulation = self._stage_to_regulation[action]
            self._pandemic_sim.impose_regulation(regulation=regulation)

        # update the sim until next regulation interval trigger and construct obs from state hist
        obs = np.array(len(self.observation_space.shape+1))

        hist_index = 0
        for i in range(self._sim_steps_per_regulation):
            # step sim
            self._pandemic_sim.step()

            # store only the last self._history_size state values
            if i >= (self._sim_steps_per_regulation - self._obs_history_size):
                unlocked_non_essential_business_locations = np.asarray([int(not cast(NonEssentialBusinessLocationState,
                                                                                        self._pandemic_sim.state.id_to_location_state[
                                                                                            loc_id]).locked)
                                                                            for loc_id in self._non_essential_business_loc_ids])

                obs[:5] = np.asarray([self._pandemic_sim.state.global_infection_summary[k] for k in self._pandemic_sim.sorted_infection_summary])[None, None, ...]

                obs[5:10] = np.asarray([self._pandemic_sim.state.global_testing_state.summary[k] for k in self._pandemic_sim.sorted_infection_summary])[None, None, ...]

                obs[10] = self._pandemic_sim.state.regulation_stage

                obs[11] = int(self._pandemic_sim.state.infection_above_threshold)

                obs[12] = int(self._pandemic_sim.state.sim_time.day)

                hist_index += 1

        prev_obs = self._last_observation
        self._last_reward = self._reward_fn.calculate_reward(prev_obs, action, obs, unlocked_non_essential_business_locations) if self._reward_fn else 0.
        done = self._done_fn.calculate_done(obs, action) if self._done_fn else False
        self._last_observation = obs

        return self._last_observation, self._last_reward, done, {}

    def reset(self) -> np.array:
        self._pandemic_sim.reset()
        self._last_observation = np.zeros(self.observation_space.shape)
        self._last_reward = 0.0
        if self._done_fn is not None:
            self._done_fn.reset()
        return self._last_observation

    def render(self, mode: str = 'human') -> bool:
        pass
