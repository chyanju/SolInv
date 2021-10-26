import time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from ray.rllib.models.torch.torch_modelv2 import TorchModelV2
from ray.rllib.utils.framework import try_import_torch
from ray.rllib.models.preprocessors import get_preprocessor
from ray.rllib.utils.annotations import override

from torch_geometric.data import Data
from torch_geometric.nn import TransformerConv

class TestNN(TorchModelV2, nn.Module):
    def __init__(self, obs_space, action_space, num_outputs, model_config, name):
        nn.Module.__init__(self)
        super().__init__(obs_space, action_space, num_outputs, model_config, name)
        # fixme: what is this? can i remove it?
        # self.obs_size = get_preprocessor(obs_space)(obs_space).size
        # self.inherited_config = model_config
        self.config = model_config["custom_model_config"]
        self.environment = self.config["environment"] # provides access to contract utils

        # embedding for all tokens
        self.token_embedding = nn.Embedding(
            num_embeddings=self.config["num_token_embeddings"],
            embedding_dim=self.config["token_embedding_dim"],
            padding_idx=self.config["token_embedding_padding_idx"],
        )

        # encoding utils for contract (igraph)
        # parameter ref: https://github.com/pyg-team/pytorch_geometric/blob/master/examples/gat.py
        self.contract_conv0 = TransformerConv(
            in_channels=self.config["token_embedding_dim"],
            out_channels=self.config["token_embedding_dim"],
            edge_dim=self.config["token_embedding_dim"],
            heads=self.config["conv_heads"], 
            dropout=self.config["conv_dropout"],
        )
        # conv1 is a helper conv that shrink the heads*dim to dim
        self.contract_conv1 = nn.Linear(
            self.config["token_embedding_dim"]*self.config["conv_heads"],
            self.config["token_embedding_dim"],
        )
        # self.contract_conv1 = TransformerConv(
        #     in_channels=self.config["token_embedding_dim"]*self.config["conv_heads"],
        #     out_channels=self.config["token_embedding_dim"],
        #     edge_dim=self.config["token_embedding_dim"],
        #     heads=1,
        #     dropout=self.config["conv_dropout"],
        #     concat=False,
        # )

        # invariant is also composed by tokens
        self.invariant_projection = nn.Linear(
            self.config["token_embedding_dim"], 
            self.config["invariant_hidden_dim"], 
        )
        self.invariant_encoder = nn.GRU(
            input_size=self.config["invariant_hidden_dim"],
            hidden_size=self.config["invariant_out_dim"],
            batch_first=True,
        )

        # helper layer for action_function
        self.action_bias_branch = nn.Linear(
            self.config["token_embedding_dim"],
            1,
        )
        self.action_projection_branch = nn.Linear(
            self.config["token_embedding_dim"],
            self.config["invariant_out_dim"],
        )
        # use default
        self.value_branch = nn.Linear(
            self.config["invariant_out_dim"],
            1,
        )

        self._invariant_features = None

        # this caches contract utils for faster switching
        # between cotnracts in training between different rollouts
        # and also prevents memory overflow from implicit tensor copy by calling
        # the type conversion methods every time (e.g., `tensor.long()`)
        self.cached_contract_utils = {}

    def where(self):
        return next(self.parameters()).device

    def recover_graph_data(self, arg_contract_id):
        # recover graph data from obs
        # note that batch size could be larger than 1 (multiple instances in a batch), need to address this

        # note: need to convert to integer here since ray encapsulates all obs in float
        # use int() since it's not for embedding
        # tmp_def: (B, 1)
        tmp_contract_id = arg_contract_id.int().cpu().numpy() 
        tmp_batch_size = tmp_contract_id.shape[0]

        data_list = []
        # process batches one by one
        # use the tensor directly
        for b in range(tmp_batch_size):

            tmp_curr_contract_id = tmp_contract_id[b, 0]
            tmp_graph = self.environment.cached_contract_utils[tmp_curr_contract_id]["contract_observed"]

            # need to get embedding
            # note: need to recover to the current device of the model
            res_graph = {
                "x": self.token_embedding(tmp_graph["x"].to(self.where())),
                "edge_attr": self.token_embedding(tmp_graph["edge_attr"].to(self.where())),
                "edge_index": tmp_graph["edge_index"].to(self.where()), # no need to embed this one
            }
            data_list.append(res_graph)

        return data_list

    @override(TorchModelV2)
    def forward(self, input_dict, state, seq_lens):
        # DEBUG
        if input_dict["obs"]["start"].shape[0]==128:
            print("\r# calling forward, B={}, ts={}".format(input_dict["obs"]["start"].shape[0], time.time()), end="")
        # state and seq_lens are not used here

        if all((input_dict["obs"]["start"].flatten() == 0).tolist()):
            # if all flags are 0, then it's for sure a dummy batch
            # print("# Model is in dummy batch mode.")
            # note: need to create tensors to the current model's device
            self._invariant_features = torch.zeros(input_dict["obs"]["start"].shape[0], self.config["invariant_out_dim"]).to(self.where())
            tmp_out = torch.zeros(input_dict["obs"]["start"].shape[0], self.config["action_out_dim"]).to(self.where())
            return tmp_out, []

        # print("# Model is in normal mode.")
        # first encode the problem graph
        # ==============================

        # tmp0_graph_data: [(B, )]
        tmp0_graph_data = self.recover_graph_data(input_dict["obs"]["contract_id"])
        # tmp1_graph_repr: [(num_nodes, token_embedding_dim), ...]
        tmp1_graph_repr = [
            F.relu(self.contract_conv1(
                F.relu(self.contract_conv0( 
                    p["x"], p["edge_index"], p["edge_attr"]
                ))
            ))
            for p in tmp0_graph_data
        ]

        # then encode the state
        # =====================

        # tmp0_inv: (B, max_step)
        tmp0_inv_token = input_dict["obs"]["action_seq@token_channel"].long()
        tmp0_inv_node = input_dict["obs"]["action_seq@node_channel"].long()

        if self.time_major:
            # (max_step, B)
            tmp0_inv_token = tmp0_inv_token.permute(1,0)
            tmp0_inv_node = tmp0_inv_node.permute(1,0)

        # tmp1_inv: (B, invariant_out_dim)
        tmp1_inv = self.encode_inv(tmp0_inv_token, tmp0_inv_node, tmp1_graph_repr)

        # (B, action_out_dim)
        tmp2_out = self.action_function(
            tmp1_inv, 
            input_dict["obs"]["all_actions@token_channel"].long(),
            input_dict["obs"]["all_actions@node_channel"].long(),
            tmp1_graph_repr,
        )

        # apply masking, ref: https://towardsdatascience.com/action-masking-with-rllib-5e4bec5e7505
        # will broadcast with shape
        inf_mask = torch.maximum( 
            torch.log(input_dict["obs"]["action_mask"]), 
            torch.tensor(torch.finfo(torch.float32).min) 
        )

        # note-important: should return state as [] because LSTM is not used between states, 
        #                 but just to encode one state, which makes the whole model behave normal
        #                 (not considering previous states)
        return tmp2_out + inf_mask, []

    def encode_inv(self, arg_inv_token, arg_inv_node, arg_graph_repr):
        # arg_inv_token, arg_inv_node: (B, max_step)

        # tmp0_inv: (B, max_step, token_embedding_dim)
        tmp0_inv = self.mixed_embedding(arg_inv_token, arg_inv_node, arg_graph_repr)

        # tmp1_inv: (B, max_step, invariant_hidden_dim)
        tmp1_inv = nn.functional.relu(self.invariant_projection(tmp0_inv))

        # tmp2_inv: (B, max_step, invariant_out_dim)
        tmp2_inv, _ = self.invariant_encoder(tmp1_inv)

        # self._invariant_features: (B, 1, invariant_out_dim) -> (B, invariant_out_dim)
        # get last time step only
        self._invariant_features = torch.reshape(
            tmp2_inv[:,-1:,:], [-1, self.config["invariant_out_dim"]]
        )

        return self._invariant_features

    def mixed_embedding(self, arg_inv_token, arg_inv_node, arg_graph_repr):
        # take over the forward method of the mixed embedding
        # for (fixed) base tokens, use the original token embedding directly
        # for flex action/token, locate to the corresponding node representation in the provided graph
        # note that batch size could be larger than 1 (multiple instances in a batch), need to address this

        # arg_inv: (B, max_step)
        # arg_graph_repr: [(num_nodes, token_embedding_dim), ...]
        tmp_batch_size = arg_inv_token.shape[0]

        result_list = []
        for b in range(tmp_batch_size):
            # note-important
            # b_inv_token: (1, max_step) if arg_inv_token is action_seq
            #                (1, len(action_list)) if arg_inv_token is all_actions
            b_inv_token = arg_inv_token[b:b+1, :]
            b_inv_node = arg_inv_node[b:b+1, :]

            # process the fixed part
            # tmp_fixed_embedding: (1, max_step, token_embedding_dim)
            tmp_fixed_embedding = self.token_embedding(b_inv_token)

            # process the flex part
            tmp_flex_embedding = F.embedding(
                input=b_inv_node, weight=arg_graph_repr[b], padding_idx=self.token_embedding.padding_idx,
                max_norm=self.token_embedding.max_norm, norm_type=self.token_embedding.norm_type,
                scale_grad_by_freq=self.token_embedding.scale_grad_by_freq, sparse=self.token_embedding.sparse,
            )

            # then add them up
            # tmp_inv_embedding: (1, max_step, token_embedding_dim)
            tmp_inv_embedding = tmp_fixed_embedding + tmp_flex_embedding
            result_list.append(tmp_inv_embedding)

        # result_embedding: (B, max_step, token_embedding_dim)
        result_embedding = torch.cat(result_list, dim=0)
        return result_embedding

    @override(TorchModelV2)
    def value_function(self):
        assert self._invariant_features is not None, "self._invariant_features is None, call forward() first."
        return torch.reshape(self.value_branch(self._invariant_features), [-1])

    def action_function(self, arg_inv, arg_aa_token, arg_aa_node, arg_graph_repr):
        # perform attention-like (dynamic activation function) over actions
        # this does not require override
        # arg_inv: (B, invariant_out_dim)
        # arg_aa_token, arg_aa_node: (B, len(env.action_list))
        # arg_graph_repr: [(num_nodes, token_embedding_dim), ...]
        B = arg_inv.shape[0]

        # tmp0: (B, len(env.action_list), token_embedding_dim)
        tmp0 = self.mixed_embedding(arg_aa_token, arg_aa_node, arg_graph_repr)
        # tmp1: (B, len(env.action_list), invariant_out_dim)
        tmp1 = self.action_projection_branch(tmp0)

        # tmp2: (B, invariant_out_dim, 1)
        tmp2 = arg_inv.view(B, self.config["invariant_out_dim"], 1)

        # simulate matrix multiplication
        # tmp3: (B, len(env.action_list), 1)
        tmp3 = torch.matmul(tmp1, tmp2)
        # tmp4: (B, len(env.action_list))
        tmp4 = tmp3.view(B, self.config["action_out_dim"])

        # tmp_bias: (B, len(env.action_list), 1)
        tmp_bias = self.action_bias_branch(tmp0).view(B, self.config["action_out_dim"])

        # apply bias
        # tmp_out: (B, len(env.action_list))
        tmp_out = tmp4 + tmp_bias

        return tmp_out

    

    

    
