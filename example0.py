import argparse
import gym
from gym.utils import seeding
from gym.spaces import Discrete, Box
import numpy as np
import os
import random
from typing import List, Any, Union, Dict

import torch

# ray related utils
import ray
from ray import tune
from ray.rllib.agents import ppo, dqn
from ray.rllib.env.env_context import EnvContext
from ray.rllib.models import ModelCatalog
from ray.rllib.models.torch.torch_modelv2 import TorchModelV2
from ray.rllib.models.torch.fcnet import FullyConnectedNetwork as TorchFC
from ray.rllib.models.torch.recurrent_net import RecurrentNetwork as TorchRNN
from ray.rllib.policy.rnn_sequencing import add_time_dimension
from ray.rllib.models.modelv2 import ModelV2
from ray.rllib.models.preprocessors import get_preprocessor
from ray.rllib.utils.annotations import override
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

class TorchRNNModel(TorchRNN, nn.Module):
    def __init__(self,
                 obs_space,
                 action_space,
                 num_outputs,
                 model_config,
                 name,
                 fc_size=64,
                 lstm_state_size=256):
        nn.Module.__init__(self)
        super().__init__(obs_space, action_space, num_outputs, model_config,
                         name)

        self.obs_size = get_preprocessor(obs_space)(obs_space).size
        self.fc_size = fc_size
        self.lstm_state_size = lstm_state_size

        # Build the Module from fc + LSTM + 2xfc (action + value outs).
        self.embedding = nn.Embedding(
            num_embeddings=model_config["custom_model_config"]["num_embeddings"],
            embedding_dim=model_config["custom_model_config"]["embedding_size"],
            padding_idx=0,
        )
        self.fc1 = nn.Linear(model_config["custom_model_config"]["embedding_size"], self.fc_size)
        self.lstm = nn.LSTM(
            self.fc_size, self.lstm_state_size, batch_first=True)
        self.action_branch = nn.Linear(self.lstm_state_size, num_outputs)
        self.value_branch = nn.Linear(self.lstm_state_size, 1)
        # Holds the current "base" output (before logits layer).
        self._features = None

    @override(ModelV2)
    def get_initial_state(self):
        # TODO: (sven): Get rid of `get_initial_state` once Trajectory
        #  View API is supported across all of RLlib.
        # Place hidden states on same device as model.
        h = [
            self.fc1.weight.new(1, self.lstm_state_size).zero_().squeeze(0),
            self.fc1.weight.new(1, self.lstm_state_size).zero_().squeeze(0)
        ]
        return h

    @override(ModelV2)
    def value_function(self):
        assert self._features is not None, "must call forward() first"
        return torch.reshape(self.value_branch(self._features), [-1])

    @override(TorchRNN)
    def forward(self, input_dict, state, seq_lens):
        # note: somewhat obs_flat is storing action_mask, so you need to use obs[inv] here
        # print("# [debug] input_dict[obs][contract] size is: {}".format(input_dict["obs"]["contract"].size()))
        # print("# [debug] input_dict[obs][inv] size is: {}".format(input_dict["obs"]["inv"].size()))
        # print("# [debug] input_dict[obs][inv] is: {}".format(input_dict["obs"]["inv"]))
        flat_inputs = input_dict["obs"]["inv"].int()
        if isinstance(seq_lens, np.ndarray):
            seq_lens = torch.Tensor(seq_lens).int()
        max_seq_len = flat_inputs.shape[0] // seq_lens.shape[0]
        self.time_major = self.model_config.get("_time_major", False)
        inputs = add_time_dimension(
            flat_inputs,
            max_seq_len=max_seq_len,
            framework="torch",
            time_major=self.time_major,
        )
        output, new_state = self.forward_rnn(inputs, state, seq_lens)
        output = torch.reshape(output, [-1, self.num_outputs])
        # apply masking, ref: https://towardsdatascience.com/action-masking-with-rllib-5e4bec5e7505
        inf_mask = torch.maximum( 
            torch.log(input_dict["obs"]["action_mask"]), 
            torch.tensor(torch.finfo(torch.float32).min) 
        )
        # print("# [debug] input_dict[obs][action_mask] is: {}".format(input_dict["obs"]["action_mask"]))
        # print("# [debug] output is: {}".format(output))
        return output + inf_mask, new_state

    @override(TorchRNN)
    def forward_rnn(self, inputs, state, seq_lens):
        """Feeds `inputs` (B x T x ..) through the Gru Unit.
        Returns the resulting outputs as a sequence (B x T x ...).
        Values are stored in self._cur_value in simple (B) shape (where B
        contains both the B and T dims!).
        Returns:
            NN Outputs (B x T x ...) as sequence.
            The state batches as a List of two items (c- and h-states).
        """
        # prev0: (B, T, 1) -> (B, T)
        prev0 = torch.flatten(inputs, start_dim=-2)
        # prev1: (B, T) -> (B, T, embd_dim)
        prev1 = self.embedding(prev0)
        # x = nn.functional.relu(self.fc1(inputs))
        # x: (B, T, embd_dim) -> (B, T, fc_size)
        x = nn.functional.relu(self.fc1(prev1))
        self._features, [h, c] = self.lstm(
            x, [torch.unsqueeze(state[0], 0),
                torch.unsqueeze(state[1], 0)])
        action_out = self.action_branch(self._features)
        # print("#[debug] action_out size is: {}".format(action_out.size()))
        return action_out, [torch.squeeze(h, 0), torch.squeeze(c, 0)]

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
    ModelCatalog.register_custom_model("my_custom_model", TorchRNNModel)

    ppo_config = ppo.DEFAULT_CONFIG.copy()
    rl_config = {
        "env": InvariantEnvironment,
        "env_config": env_config,
        "model": {
            "custom_model": "my_custom_model",
            "custom_model_config": {
                "embedding_size": 16,
                "num_embeddings": len(tmp_environment.token_list),
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
