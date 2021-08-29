import argparse
import gym
from gym.utils import seeding
from gym.spaces import Discrete, Box
import numpy as np
import os
import random
from typing import List, Any, Union, Dict

# ray related utils
import ray
from ray import tune
from ray.rllib.agents import ppo
from ray.rllib.env.env_context import EnvContext
from ray.rllib.models import ModelCatalog
from ray.rllib.models.torch.torch_modelv2 import TorchModelV2
from ray.rllib.models.torch.fcnet import FullyConnectedNetwork as TorchFC
from ray.rllib.utils.framework import try_import_torch
from ray.rllib.utils.test_utils import check_learning_achieved
from ray.tune import grid_search
from ray.tune.logger import pretty_print
torch, nn = try_import_torch()

# trinity related utils
import solinv.tyrell.spec as S
import solinv.tyrell.dsl as D
from solinv.tyrell.interpreter import InvariantInterpreter
from solinv.tyrell.dsl import Node, HoleNode
from solinv.tyrell.dsl.utils import derive_dfs, get_hole_dfs
from solinv.environment import InvariantEnvironment

if __name__ == "__main__":
    spec = S.parse_file("./dsls/example0.tyrell")
    start_type = spec.get_type("Expr")
    interpreter = InvariantInterpreter()
    env_config = {
        "spec": spec,
        "start_type": start_type,
        "max_step": 10,
        "contract_path": "/Users/joseph/Desktop/UCSB/21fall/SolidTypes/test/regression/good/mint_MI.sol",
        "interpreter": interpreter
    }
    # environment = InvariantEnvironment(config=env_config)

    ray.init(local_mode=True)
    rl_config = ppo.DEFAULT_CONFIG.copy()
    rl_config["num_workers"] = 1
    rl_config["num_sgd_iter"] = 50
    rl_config["sgd_minibatch_size"] = 32
    rl_config["model"]["fcnet_hiddens"] = [64,64]
    rl_config["env_config"] = env_config

    agent = ppo.PPOTrainer(rl_config, env=InvariantEnvironment)
    for i in range(100):
        res = agent.train()
        # print("# episode_reward_mean: {}".format(res["episode_reward_mean"]))
        print(pretty_print(res))

    ray.shutdown()
