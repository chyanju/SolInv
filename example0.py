import argparse
import numpy as np
import os
import random

# ray related utils
import ray
from ray import tune
from ray.rllib.agents import ppo, dqn
from ray.rllib.models import ModelCatalog
from ray.tune.logger import pretty_print

# trinity related utils
import solinv.tyrell.spec as S
import solinv.tyrell.dsl as D
from solinv.tyrell.interpreter import InvariantInterpreter
from solinv.environment import InvariantEnvironment
from solinv.model import InvariantGRU, SimpleInvariantGRU

if __name__ == "__main__":
    spec = S.parse_file("./dsls/abstract0.tyrell")
    start_type = spec.get_type("Expr")
    interpreter = InvariantInterpreter()
    env_config = {
        "spec": spec,
        "start_type": start_type,
        "max_step": 6,
        "contract_path": "../SolidTypes/test/regression/good/mint_MI.sol",
        "max_contract_length": 2500,
        # options are: 0.4.26, 0.5.17, 0.6.12
        "solc_version": "0.5.17",
        "token_list_path": "./token_list0.pkl",
        "interpreter": interpreter
    }
    # need to construct the vocab first to provide parameters for nn
    tmp_environment = InvariantEnvironment(config=env_config)

    ray.init(local_mode=True)
    # ModelCatalog.register_custom_model("invariant_gru", InvariantGRU)
    ModelCatalog.register_custom_model("invariant_gru", SimpleInvariantGRU)

    ppo_config = ppo.DEFAULT_CONFIG.copy()
    rl_config = {
        "env": InvariantEnvironment,
        "env_config": env_config,
        "model": {
            "custom_model": "invariant_gru",
            "custom_model_config": {
                "num_embeddings": len(tmp_environment.token_list),
                "embedding_size": 16,

                "invariant_projection_size": 64,
                "invariant_encoder_state_size": 64,

                "problem_projection_size": 64,
                "problem_encoder_state_size": 64,

                "output_size": len(tmp_environment.action_list),
            },
        },
        "num_workers": 1,
        "framework": "torch",
    }
    ppo_config.update(rl_config)
    agent = ppo.PPOTrainer(env=InvariantEnvironment, config=ppo_config)
    
    for i in range(100):
        res = agent.train()
        # print("# episode_reward_mean: {}".format(res["episode_reward_mean"]))
        print(pretty_print(res))

    ray.shutdown()
