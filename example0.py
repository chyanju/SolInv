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
from solinv.model import InvariantTGN, InvariantGCN

if __name__ == "__main__":
    spec = S.parse_file("./dsls/abstract0.tyrell")
    start_type = spec.get_type("Expr")
    interpreter = InvariantInterpreter()
    env_config = {
        "spec": spec,
        "start_type": start_type,
        "max_step": 6,
        # version options are: 0.4.26, 0.5.17, 0.6.12
        "contracts": [
            # sum(_balances) <= _totalSupply
            # ("./benchmarks/mint_MI.sol", "0.5.17"),

            # 1. sum(balances) <= totalSupply_
            # liquidsol-exe ./benchmarks/easy/0x28b5e12cce51f15594b0b91d5b5adaa70f684a02.sol --task check --check-inv 'sum(balances) <= totalSupply_' --only-last
            ("./benchmarks/easy/0x28b5e12cce51f15594b0b91d5b5adaa70f684a02.sol", "0.4.26"), # stovars: 9

            # 2. sum(balances) <= totalSupply_
            # liquidsol-exe ./benchmarks/easy/0x6704b673c70de9bf74c8fba4b4bd748f0e2190e1.sol --task check --check-inv 'sum(balances) <= totalSupply_' --only-last
            ("./benchmarks/easy/0x6704b673c70de9bf74c8fba4b4bd748f0e2190e1.sol", "0.4.26"), # stovars: 8

            # 3. sum(balances) <= totalSupply
            # liquidsol-exe ./benchmarks/easy/0x6745fab6801e376cd24f03572b9c9b0d4edddccf.sol --task check --check-inv 'sum(balances) <= totalSupply' --only-last
            ("./benchmarks/easy/0x6745fab6801e376cd24f03572b9c9b0d4edddccf.sol", "0.4.26"), # stovars: 8

            # 4. sum(balances) <= totalSupply
            # liquidsol-exe ./benchmarks/easy/0x9041fe5b3fdea0f5e4afdc17e75180738d877a01.sol --task check --check-inv 'sum(balances) <= totalSupply' --only-last
            ("./benchmarks/easy/0x9041fe5b3fdea0f5e4afdc17e75180738d877a01.sol", "0.4.26"), # stovars: 7

            # ("./benchmarks/easy/0x5e6016ae7d7c49d347dcf834860b9f3ee282812b.sol", "0.4.26"), # stovars: 21
            # ("./benchmarks/easy/0x286BDA1413a2Df81731D4930ce2F862a35A609fE.sol", "0.4.26"), # stovars: 11

            # fixme: NotImplementedError: Unsupported nodeType, got: UserDefinedTypeName.
            # ("./benchmarks/easy/0x888666CA69E0f178DED6D75b5726Cee99A87D698.sol", "0.4.26"),
        ],
        "interpreter": interpreter
    }
    # need to construct the vocab first to provide parameters for nn
    tmp_environment = InvariantEnvironment(config=env_config)

    # ray.init(local_mode=True)
    # use non local mode to enable GPU
    ray.init()
    # ModelCatalog.register_custom_model("invariant_tgn", InvariantTGN)
    ModelCatalog.register_custom_model("invariant_gcn", InvariantGCN)

    rl_config = ppo.DEFAULT_CONFIG.copy()
    rl_config = {
        "env": InvariantEnvironment,
        "env_config": env_config,
        "model": {
            "custom_model": "invariant_gcn",
            "custom_model_config": {
                "num_token_embeddings": len(tmp_environment.token_list),
                "token_embedding_dim": 16,
                "token_embedding_padding_idx": tmp_environment.token_dict["<PAD>"],

                "invariant_hidden_dim": 64,
                "invariant_out_dim": 32,

                "action_out_dim": len(tmp_environment.action_list),

                # this provides a share to access the contract related utils
                "environment": InvariantEnvironment,
            },
        },
        "num_workers": 1,
        "num_gpus": 1,
        "framework": "torch",
    }

    # tune.run("PPO", stop={"episode_reward_mean": 200}, config=rl_config)

    agent = ppo.PPOTrainer(env=InvariantEnvironment, config=rl_config)
    
    for i in range(100):
        print("# i={}".format(i))
        res = agent.train()
        print(pretty_print(res))

    ray.shutdown()
