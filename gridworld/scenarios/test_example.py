"""
This example is modified from the original OpenAI MADDPG training script.
Original script can be found below:
https://github.com/openai/maddpg/blob/master/experiments/train.py

To run this example, please make sure you have the OpenAI MADDPG installed,
see README under examples/marl/openai for details.

"""

import argparse
import json
import logging
import os
import random
import uuid

from datetime import datetime

import numpy as np
#import tensorflow as tf
import time
import pickle

#import maddpg.common.tf_util as U
#from maddpg.trainer.maddpg import MADDPGAgentTrainer
#import tensorflow.contrib.layers as layers

from gridworld.log import logger
from gridworld.multiagent_env import MultiAgentEnv
from gridworld.multiagent_list_interface_env import MultiAgentListInterfaceEnv
from gridworld.scenarios.buildings import make_env_config

logger.setLevel(logging.ERROR)


class CoordinatedMultiBuildingControlEnv(MultiAgentEnv):
    """ Extend the original multiagent environment to include coordination.
    In addition to the original agent-level reward, grid-level reward/penalty
    is considered, if agents fail to coordinate to satisfy the system-level
    constraint(s).
    In this example, we consider the voltage constraints: agents need to
    coordinate so the common bus voltage is within the ANSI C.84.1 limit.
    Otherwise, the voltage violation penalty will be shared by all agents.
    """

    VOLTAGE_LIMITS = [0.95, 1.05]
    VV_UNIT_PENALTY = 1e4

    # Overwriting the default transform behavior.
    def reward_transform(self, rew_dict) -> dict:
        """ Adding system wide penalty and slip it evenly on all agents.
        """

        voltage_violation = self.get_voltage_violation()
        sys_penalty = voltage_violation * self.VV_UNIT_PENALTY

        # split the penalty equally among agents.
        agent_num = len(rew_dict)
        for key in rew_dict.keys():
            rew_dict[key] -= (sys_penalty / agent_num)

        return rew_dict

    def meta_transform(self, meta) -> dict:
        """ Augment meta info for logging purpose. """
        sys_info = {'voltage_violation': self.get_voltage_violation()}
        meta.update(sys_info)
        return meta

    def get_voltage_violation(self):
        """ Obtain voltage of the bus where all buildings connect to.
        """

        assert len(set(
            self.agent_name_bus_map.values())) == 1, \
            "In this example, all buildings should be on the same bus."

        bus_id = list(set(self.agent_name_bus_map.values()))[0]

        common_bus_voltage = self.pf_solver.get_bus_voltage_by_name(bus_id)
        voltage_violation = max([
            0.0,
            self.VOLTAGE_LIMITS[0] - common_bus_voltage,
            common_bus_voltage - self.VOLTAGE_LIMITS[1]
        ])

        return voltage_violation

def make_env():
        
    """ Make the example coordinated multi-building environment.
    """

    env_config = make_env_config(
        building_config={},
        pv_config={
            "profile_csv": "pv_profile.csv",
            "scaling_factor": 40.
        },
        storage_config={
            "max_power": 15.,
            "storage_range": (3., 50.)
        },
        system_load_rescale_factor=0.6, # arglist.sys_load,
        num_buildings=3
    )

    env = MultiAgentListInterfaceEnv(
        CoordinatedMultiBuildingControlEnv,
        env_config
    )
    
    import ipdb
    # ipdb.set_trace()
    
    env.n_agents = env.n
    env.n_constraints = 0

    return env

import gym
gym.register(
    id='MultiBuilding-v0',
    entry_point='gridworld.scenarios.test_example:make_env',
)