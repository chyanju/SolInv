import re
import copy
import json
import pickle
import subprocess
import torch
import numpy as np
from typing import List, Any, Union, Dict

import gym
import igraph
from gym.utils import seeding

from ..tyrell import spec as S
from ..tyrell import dsl as D
from ..tyrell.interpreter import InvariantInterpreter
from ..tyrell.dsl import Node, HoleNode
from ..tyrell.dsl.utils import derive_dfs, get_hole_dfs

from .invariant_heuristic import InvariantHeuristic
from .error import EnvironmentError

# import torch
# class Tensor(torch.Tensor):
#     @staticmethod
#     def __new__(cls, *args, info=None, **kwargs):
#         return super().__new__(cls, *args, **kwargs)
    
#     def __init__(self, *args, info=None, **kwargs):
#         super().__init__() # optional
#         self.info = info

class InvariantEnvironment(gym.Env):
    # note: class static variable
    #       used to track previously sampled sequences, for coverage based exploration
    sampled_action_seqs = {}
    CONTRACT_MAX_NODES = 10000
    CONTRACT_MAX_EDGES = 10000

    def __init__(self, config: Dict[str, Any]):
        self.tspec = config["spec"]
        self.builder = D.Builder(self.tspec)
        self.start_type = config["start_type"]
        self.max_step = config["max_step"]
        self.solc_version = config["solc_version"]
        self.interpreter = config["interpreter"]
        self.contract_path = config["contract_path"]

        # ================== #
        # vocabulary related #
        # ================== #
        # action list that contains every production rule in the dsl
        self.action_list = self.tspec.productions()
        self.action_dict = {self.action_list[i]:i for i in range(len(self.action_list))}
        # a fixed action is shared across different benchmarks
        self.fixed_action_list = list(filter(
            lambda x: not(x.is_enum() and "<VAR" in x._get_rhs()),
            self.action_list,
        ))
        self.fixed_action_dict = {self.fixed_action_list[i]:i for i in range(len(self.fixed_action_list))}
        # a flex action is bounded with a stovar dynamically for different benchmarks
        self.flex_action_list = list(filter(
            lambda x: x not in self.fixed_action_list,
            self.action_list,
        ))
        self.flex_action_dict = {self.flex_action_list[i]:i for i in range(len(self.flex_action_list))}
        # note: re-order action list to have fixed+flex order
        self.action_list = self.fixed_action_list + self.flex_action_list
        self.action_dict = {self.action_list[i]:i for i in range(len(self.action_list))}

        self.special_token_list = ["<PAD>", "<ID>", "<REF>"]
        self.reserved_identifier_token_list = ["require"]
        self.reserved_vertex_token_list = sorted([
            "<DICT>", "<LIST>",
            "!=", "+", "=", ">=",
            "Mapping", "uint256",
            "number"
        ])
        self.reserved_edge_token_list = sorted([
            0, 1, 2, 3, 4, 5, 6, 7, 8, 9
        ]) + sorted([
            "arguments", "operator", "leftExpression", "rightExpression", "expression", "leftHandSide", "rightHandSide",
            "baseExpression", "indexExpression", "topType", "keyType", "valueType"
        ])
        # every token in the token list should have a fixed embedding for
        self.base_token_list = self.special_token_list \
                             + self.reserved_identifier_token_list \
                             + self.reserved_vertex_token_list \
                             + self.reserved_edge_token_list
        self.token_list = self.base_token_list + self.fixed_action_list
        self.token_dict = {self.token_list[i]:i for i in range(len(self.token_list))}

        # ================ #
        # contract related #
        # ================ #
        # tokenize the target contract
        self.contract_json = self.get_contract_ast(self.contract_path, self.solc_version)
        self.contract_static_env, self.contract_slim_ast = self.get_slim_ast(self.contract_json)
        # e2n: variable name -> node id
        #      e.g., {'_balances': 0, '_totalSupply': 4, 'account': 5, 'value': 6}
        # e2r: variable name -> list of node ids that refer to this variable, e.g., 
        #      {'account': [13, 37, 42],
        #       '_totalSupply': [22, 24, 28, 31],
        #       'value': [23, 32, 43],
        #       '_balances': [36, 41]}
        # - an variable name is NOT always a stovar
        # - an identifier is NOT always a variable (it could also be some reserved one like "require")
        self.contract_e2n, self.contract_e2r, self.contract_igraph, self.contract_root_id = self.slim_ast_to_igraph(self.contract_static_env, self.contract_slim_ast)
        # self.contract_networkx = igraph.Graph.to_networkx(self.contract_igraph)
        # map tokens to corresponding ids (no variable will show up since the graph is already anonymous)
        self.contract_encoded_igraph = self.contract_igraph.copy()
        for p in self.contract_encoded_igraph.vs:
            p["token"] = self.token_dict[p["token"]]
        for p in self.contract_encoded_igraph.es:
            p["token"] = self.token_dict[p["token"]]
        self.contract_observed = {
            "def": [len(self.contract_encoded_igraph.vs), len(self.contract_encoded_igraph.es)],
            "x": self.pad_to_length(self.contract_encoded_igraph.vs["token"], InvariantEnvironment.CONTRACT_MAX_NODES),
            "edge_attr": self.pad_to_length(self.contract_encoded_igraph.es["token"], InvariantEnvironment.CONTRACT_MAX_EDGES),
            "edge_index_src": self.pad_to_length([self.contract_encoded_igraph.es[i].source for i in range(len(self.contract_encoded_igraph.es))], InvariantEnvironment.CONTRACT_MAX_EDGES),
            "edge_index_tgt": self.pad_to_length([self.contract_encoded_igraph.es[i].target for i in range(len(self.contract_encoded_igraph.es))], InvariantEnvironment.CONTRACT_MAX_EDGES),
        }

        # get stovars list
        self.stovar_list = self.get_contract_stovars(self.contract_path)
        self.stovar_dict = {self.stovar_list[i]:i for i in range(len(self.stovar_list))}
        # check for enough var production rules in the dsl
        for i in range(len(self.stovar_list)):
            _ = self.tspec.get_enum_production_or_raise(self.tspec.get_type("EnumExpr"), "<VAR{}>".format(i))
        # establish the flex-stovar bindings
        self.flex_action_to_stovar = {
            self.tspec.get_enum_production_or_raise(self.tspec.get_type("EnumExpr"), "<VAR{}>".format(i)) : self.stovar_list[i]
            for i in range(len(self.stovar_list))
        }
        self.stovar_to_flex_action = { self.flex_action_to_stovar[dkey]:dkey for dkey in self.flex_action_to_stovar.keys() }

        # ====== #
        # basics #
        # ====== #
        # initialize internal variables
        self.curr_trinity_inv = None # invariant in trinity node structure
        # action_seq: represented using ids from action_list, for internal tracking of the environment
        self.curr_action_seq = None
        _ = self.reset()

        # inherited variables
        # note: for action space, we are using the maximum available productions
        #       in practice, we may be just using a subset of them, i.e., with some of the flex action not used
        self.action_space = gym.spaces.Discrete(len(self.action_list))
        self.observation_space = gym.spaces.Dict({
            "start": gym.spaces.Box(0,1,shape=(1,),dtype=np.int32),
            "contract": gym.spaces.Dict({
                # specifies: [num_nodes, num_edges]
                "def": gym.spaces.Box(
                    0, max(InvariantEnvironment.CONTRACT_MAX_NODES, InvariantEnvironment.CONTRACT_MAX_EDGES), 
                    shape=(2,), dtype=np.int32
                ),
                # (num_edges,): node features, every node has 1 token as feature
                "x": gym.spaces.Box(0, len(self.token_list)-1, shape=(InvariantEnvironment.CONTRACT_MAX_NODES,), dtype=np.int32),
                # valid shape (num_edges,): edge features, every edge has 1 token as feature
                "edge_attr": gym.spaces.Box(0, len(self.token_list)-1, shape=(InvariantEnvironment.CONTRACT_MAX_EDGES,), dtype=np.int32),
                # valid shape (num_edges,): edge definitions - source
                "edge_index_src": gym.spaces.Box(0, InvariantEnvironment.CONTRACT_MAX_NODES-1, shape=(InvariantEnvironment.CONTRACT_MAX_EDGES,), dtype=np.int32),
                # valid shape (num_edges,): edge definitions - target
                "edge_index_tgt": gym.spaces.Box(0, InvariantEnvironment.CONTRACT_MAX_NODES-1, shape=(InvariantEnvironment.CONTRACT_MAX_EDGES,), dtype=np.int32),
            }),
            "action_mask": gym.spaces.Box(0, 1, shape=(len(self.action_list),), dtype=np.int32), # for output layer, no need to + len(sptok_list)
            # fixme: 1000 should be MAX_NODES
            "nn_seq": gym.spaces.Box(0, len(self.token_list)+1000, shape=(self.max_step, ), dtype=np.int32), # for encoding layer, need to + len(sptok_list)
            "action_seq": gym.spaces.Box(0, len(self.token_list)+1000, shape=(len(self.action_list),), dtype=np.int32), # for dynamic action output, remain the same for the same contract
        })

    def action_seq_to_nn_seq(self, arg_action_seq):
        # nn_seq: represented using ids from (overflow) embedding, for internal tracking of the neural network
        # orderings:
        #   [ inflow                      | overflow             ]
        #   [ token list                  | node id space        ]
        #   [ base tokens | fixed actions | node id space        ]
        #   [ off-the-shelf embeddings    | aggregated embedings ]
        # for node id part, node_id = nn_id - len(token_list), i.e., the remaining (overflow part) is the id
        ret_seq = []
        for p in arg_action_seq:
            if p >= len(self.token_list):
                # flex action
                ret_seq.append(
                    self.contract_e2n[self.flex_action_to_stovar[self.token_list[p]]] + len(self.token_list)
                )
            else:
                # fixed action
                ret_seq.append(p+len(self.base_token_list))
        return ret_seq

    def pad_to_length(self, arg_obj, arg_length):
        return arg_obj + [self.token_dict["<PAD>"] for _ in range(arg_length-len(arg_obj))]

    def trinity_inv_to_debugging_inv(self, arg_trinity_inv):
        # debugging inv still retains the recursive trinity structure
        # this will replace all replacable <VAR?> with binded stovar
        tmp_inv = str(arg_trinity_inv)
        for prod in self.flex_action_to_stovar.keys():
            tmp_inv = tmp_inv.replace(prod._get_rhs(), self.flex_action_to_stovar[prod])
        return tmp_inv

    def trinity_inv_to_verifier_inv(self, arg_trinity_inv):
        # verifier inv will be the string that is diredtly provided to the verifier
        tmp_inv0 = self.interpreter.eval(arg_trinity_inv)
        tmp_inv1 = self.trinity_inv_to_debugging_inv(tmp_inv0) # compatible reuse
        return tmp_inv1

    def record_action_seq(self, arg_action_seq):
        # add the sequence to the class level recorder and count
        tup_seq = tuple(arg_action_seq)
        if tup_seq not in InvariantEnvironment.sampled_action_seqs.keys():
            InvariantEnvironment.sampled_action_seqs[tup_seq] = 0
        InvariantEnvironment.sampled_action_seqs[tup_seq] += 1

    def get_contract_ast(self, arg_path, arg_solc_version):
        cmd_set = subprocess.run("solc-select use {}".format(arg_solc_version), shell=True, capture_output=True)
        if cmd_set.returncode != 0:
            raise Exception("Error executing solc-select. Check your environment configuration.")

        cmd_solc = subprocess.run("solc {} --ast-compact-json".format(arg_path), shell=True, capture_output=True)
        if cmd_solc.returncode != 0:
            raise Exception("Error executing solc <contract_path> --ast-compact-json. Check your environment configuration.")

        raw_output = cmd_solc.stdout.decode("utf-8")
        # strip out the irrelevant part
        tmp_pos = raw_output.index("=======\n") # raise if not found
        raw_json = raw_output[tmp_pos+len("=======\n"):]
        parsed_json = json.loads(raw_json)
        return parsed_json

    def get_contract_stovars(self, arg_path):
        cmd_sto = subprocess.run("liquidsol-exe {} --task stovars".format(arg_path), shell=True, capture_output=True)
        assert(cmd_sto.returncode == 0)
        raw_output = cmd_sto.stdout.decode("utf-8")
        lines = raw_output.rstrip().split("\n")[1:]
        assert(len(lines) % 2 == 0)
        n = len(lines) // 2
        tmp_list = []
        for i in range(n):
            tmp_list.append(lines[i])
        return tmp_list

    # fixme: need to adjust the extraction method to account for different contracts
    # note: currently the whole environment assumes globally unique variable/identifier name
    #       i.e., all variables can only be defined once, regardless of any scope
    def get_slim_ast(self, arg_json):
        # start from a specific "ContractDefinition" node
        start_json = arg_json["nodes"][0]
        static_env = {}
        return_json = self._rec_extract_slim_ast(start_json, static_env)
        return (static_env, return_json)

    # venv: virtual environemnt that provies access to in-scope variable declarations
    def _rec_extract_slim_ast(self, arg_json, inherited_venv):
        ret_obj = None
        if isinstance(arg_json, dict):
            if "nodeType" in arg_json.keys():
                if arg_json["nodeType"] == "ContractDefinition":
                    ret_obj = [self._rec_extract_slim_ast(p, inherited_venv) for p in arg_json["nodes"]]
                elif arg_json["nodeType"] == "VariableDeclaration":
                    tmp_name = arg_json["name"]
                    if arg_json["typeName"]["nodeType"] == "Mapping":
                        tmp_type = {
                            "topType": "Mapping",
                            "keyType": arg_json["typeName"]["keyType"]["name"],
                            "valueType": arg_json["typeName"]["valueType"]["name"],
                        }
                    elif arg_json["typeName"]["nodeType"] == "ElementaryTypeName":
                        tmp_type = arg_json["typeName"]["name"]
                    else:
                        raise NotImplementedError("Unsupported VariableDeclaration, got: {}.".format(arg_json))
                    # directly override even if there's already on in inherited env
                    inherited_venv[tmp_name] = tmp_type
                elif arg_json["nodeType"] == "FunctionDefinition":
                    # first set the parameters' types into local env
                    for p in arg_json["parameters"]["parameters"]:
                        self._rec_extract_slim_ast(p, inherited_venv) 
                    ret_obj = self._rec_extract_slim_ast(arg_json["body"], inherited_venv)
                elif arg_json["nodeType"] == "Block":
                    ret_obj = [self._rec_extract_slim_ast(p, inherited_venv) for p in arg_json["statements"]]
                elif arg_json["nodeType"] == "ExpressionStatement":
                    ret_obj = self._rec_extract_slim_ast(arg_json["expression"], inherited_venv)
                elif arg_json["nodeType"] == "FunctionCall":
                    ret_obj = {
                        "arguments": [self._rec_extract_slim_ast(p, inherited_venv) for p in arg_json["arguments"]],
                        "expression": self._rec_extract_slim_ast(arg_json["expression"], inherited_venv),
                    }
                elif arg_json["nodeType"] == "Identifier":
                    if arg_json["name"] in self.reserved_identifier_token_list:
                        # use the original value
                        ret_obj = arg_json["name"]
                    else:
                        ret_obj = ("<IDENTIFIER>", arg_json["name"])
                elif arg_json["nodeType"] == "Literal":
                    if arg_json["kind"] == "number":
                        # ret_obj = arg_json["value"]
                        ret_obj = "number"
                    else:
                        raise NotImplementedError("Unsupported literal, got: {}.".format(arg_json))
                elif arg_json["nodeType"] == "BinaryOperation":
                    ret_obj = {
                        "operator": arg_json["operator"],
                        "leftExpression": self._rec_extract_slim_ast(arg_json["leftExpression"], inherited_venv),
                        "rightExpression": self._rec_extract_slim_ast(arg_json["rightExpression"], inherited_venv),
                    }
                elif arg_json["nodeType"] == "Assignment":
                    ret_obj = {
                        "operator": arg_json["operator"],
                        "leftHandSide": self._rec_extract_slim_ast(arg_json["leftHandSide"], inherited_venv),
                        "rightHandSide": self._rec_extract_slim_ast(arg_json["rightHandSide"], inherited_venv),
                    }
                elif arg_json["nodeType"] == "IndexAccess":
                    ret_obj = {
                        "baseExpression": self._rec_extract_slim_ast(arg_json["baseExpression"], inherited_venv),
                        "indexExpression": self._rec_extract_slim_ast(arg_json["indexExpression"], inherited_venv),
                    }
                else:
                    raise NotImplementedError("Unsupported nodeType, got: {}.".format(arg_json["nodeType"]))
            else:
                raise NotImplementedError("Unsupported dictionary.")
        else:
            raise NotImplementedError("Unsupported type of json object, got: {}.".format(type(arg_json)))
        
        # determine whether the current object is worth returning or not
        if ret_obj is None:
            return None
        elif isinstance(ret_obj, list):
            ret_obj = list(filter(lambda x: x is not None, ret_obj))
            if len(ret_obj)>0:
                return ret_obj
            else:
                return None
        elif isinstance(ret_obj, dict):
            return ret_obj
        elif isinstance(ret_obj, str):
            return ret_obj
        elif isinstance(ret_obj, tuple):
            # for debugging: identifier type replacement
            return ret_obj
        else:
            raise NotImplementedError("You should not have reached here; ret_obj type is: {}.".format(type(ret_obj)))

    def slim_ast_to_igraph(self, arg_env, arg_slim_ast):
        vtl, el, etl = [], [], []
        # first prepare nodes in env also as igraph nodes
        env_id_to_node_id = {} # e2n
        env_id_to_ref_ids = {} # e2r
        for dkey in arg_env.keys():
            # tmp_id0 is the top/root node
            tmp_id0 = self._rec_construct_edges_and_vertices(env_id_to_node_id, env_id_to_ref_ids, arg_env[dkey], vtl, el, etl)
            env_id_to_node_id[dkey] = tmp_id0
        
        # then prepare nodes in slim ast
        root_id = self._rec_construct_edges_and_vertices(env_id_to_node_id, env_id_to_ref_ids, arg_slim_ast, vtl, el, etl)
        tmp_graph = igraph.Graph(
            directed=True,
            n=len(vtl), vertex_attrs={"token":vtl},
            edges=el, edge_attrs={"token": etl},
        )
        return (env_id_to_node_id, env_id_to_ref_ids, tmp_graph, root_id)

    # this oeprates in place
    # returns the current vertex id
    def _rec_construct_edges_and_vertices(self, e2n, e2r, arg_slim_ast, vertex_token_list, edge_list, edge_token_list):
        if isinstance(arg_slim_ast, dict):
            tmp_vid0 = len(vertex_token_list)
            vertex_token_list.append("<DICT>") # special token <DICT>
            for dkey in arg_slim_ast.keys():
                dcid = self._rec_construct_edges_and_vertices(
                    e2n, e2r, arg_slim_ast[dkey], vertex_token_list, edge_list, edge_token_list
                )
                edge_token_list.append(dkey)
                edge_list.append((tmp_vid0, dcid))
                # reverse link (add if needed)
                # edge_token_list.append(dkey)
                # edge_list.append((dcid, tmp_vid0))
            return tmp_vid0
        elif isinstance(arg_slim_ast, list):
            tmp_vid0 = len(vertex_token_list)
            vertex_token_list.append("<LIST>") # special token <LIST>
            children_vertices = [
                self._rec_construct_edges_and_vertices(e2n, e2r, arg_slim_ast[i], vertex_token_list, edge_list, edge_token_list)
                for i in range(len(arg_slim_ast))
            ]
            for i in range(len(children_vertices)):
                edge_token_list.append(i) # pure number index
                edge_list.append((tmp_vid0, children_vertices[i])) # curr -> child
                # reverse link (add if needed)
                # edge_token_list.append(i)
                # edge_list.append((i, tmp_vid0))
            return tmp_vid0
        elif isinstance(arg_slim_ast, str):
            tmp_vid0 = len(vertex_token_list)
            vertex_token_list.append(arg_slim_ast) # use its own value
            # no children to add, just return
            return tmp_vid0
        elif isinstance(arg_slim_ast, tuple):
            if arg_slim_ast[0] ==  "<IDENTIFIER>":
                # seek from e2n and point to its real structure
                assert arg_slim_ast[1] in e2n.keys(), "Identifier {} is not in e2n!".format(arg_slim_ast[1])

                tmp_vid0 = len(vertex_token_list)
                vertex_token_list.append("<ID>") # use a special value
                edge_token_list.append("<REF>")
                edge_list.append((tmp_vid0, e2n[arg_slim_ast[1]]))

                # track the full list of node ids that bind to corresponding env ids
                if arg_slim_ast[1] not in e2r.keys():
                    e2r[arg_slim_ast[1]] = []
                e2r[arg_slim_ast[1]].append(tmp_vid0)

                return tmp_vid0
            else:
                raise NotImplementedError("Unsupported tuple, got: {}.".format(arg_slim_ast))
        else:
            raise NotImplementedError("Unsupported type of json object, got: {}.".format(arg_slim_ast))

    def get_action_mask(self, arg_type):
        # get action mask that allows for a specific type
        return [1 if arg_type == p.lhs else 0 for p in self.action_list]

    def is_max(self):
        '''
        Returns whether or not the length of action sequence has already reached the preset limit.
        '''
        # -1 since we always have a <SOS> at the beginning
        return len(self.curr_action_seq) >= self.max_step

    def is_done(self):
        '''
        Returns whether or not there's still any remaining hole in the current invariant.
        '''
        next_hole = get_hole_dfs(self.curr_trinity_inv)
        if next_hole is None:
            return True
        else:
            return False
    
    def reset(self):
        # note: this should return data structure as defined by self.observation_space
        #       not only the state (but also including any action mask)
        self.curr_trinity_inv = HoleNode(type=self.start_type)
        self.curr_action_seq = []
        return {
            "start": [1],
            "contract": self.contract_observed,
            "action_mask": self.get_action_mask(self.start_type),
            "nn_seq": self.pad_to_length(self.action_seq_to_nn_seq(self.curr_action_seq), self.max_step),
            "action_seq": self.action_seq_to_nn_seq(list(range(len(self.action_list)))),
        }

    def check(self, arg_contract_path: str, arg_verifier_inv: str):
        ret = subprocess.run(
            "liquidsol-exe {} --task check --check-inv '{}'".format(arg_contract_path, arg_verifier_inv),
            shell=True, capture_output=True,
        )
        if ret.returncode != 0:
            return None
        ret = ret.stdout.decode("utf-8")
        hard_ok, hard = re.search("Hard: ([0-9]+) / ([0-9]+)", ret).groups()
        soft_ok, soft = re.search("Soft: ([0-9]+) / ([0-9]+)", ret).groups()
        print()
        print("# [debug] --------------------------------------------- checking good, result: {} ---------------------------------------------".format([hard_ok, hard, soft_ok, soft]))
        print()
        result = list(map(int, [hard_ok, hard, soft_ok, soft]))
        if result[0]+result[2]==result[1]+result[3]:
            input("Found the ground truth!")
        return result

    # the action id here is for the action_list / action space for sure    
    def step(self, arg_action_id: int):
        '''
        returns: [state, reward, terminate, info]
        '''
        if arg_action_id >= len(self.action_list):
            raise EnvironmentError("Action id is not in range, required: [0, {}), got: {}".format(len(self.action_list), arg_action_id))
        
        if self.is_done():
            raise EnvironmentError("the invariant is already complete; no action is required.")

        # perform the action: derive method will raise exceptions by itself
        try:
            sts, new_inv = derive_dfs(self.builder, self.curr_trinity_inv, self.action_list[arg_action_id])
        except:
            # Exception: Types don't match, expect Empty, got Expr
            self.curr_action_seq = self.curr_action_seq + [arg_action_id]
            print("# [debug][done/exception] seq: {}, inv(before): {}".format(
                self.curr_action_seq, self.trinity_inv_to_debugging_inv(self.curr_trinity_inv),
            ))
            return [
                {
                    # you can't fill any hole since the seq terminates with an exception
                    "start": [1],
                    "contract": self.contract_observed,
                    "action_mask": [0 for _ in range(len(self.action_list))], 
                    "nn_seq": self.pad_to_length(self.action_seq_to_nn_seq(self.curr_action_seq), self.max_step),
                    "action_seq": self.action_seq_to_nn_seq(list(range(len(self.action_list)))),
                }, 
                0.0, # reward 
                True, # terminate
                {}, # info
            ]

        if not sts:
            raise EnvironmentError("node is not expanded, check the implementation")
        # if you are here, then the derivation is successful
        # then refresh state
        self.curr_trinity_inv = new_inv
        self.curr_action_seq = self.curr_action_seq + [arg_action_id]
        
        # ================================ #
        # ====== reward computation ====== #
        # ================================ #
        # hm: heuristic multiplier (default 1.0, any heuristic failing will make it 0.1)
        # rm: repeat multiplier (default 1.0, computed by 1.0/<times>)
        # all rewards will be multiplied by hm and rm
        # there are different cases
        # if the invariant is complete
        #   - if it fails some heuristics: 1.0
        #   - else
        #     - if it fails the checking: 0.1
        #     - if it passes the checking: 10.0 * percentage_of_constraints_passed 
        # if the invariant is not complete
        #   - but it reaches the max allowed step: 0.0 (which means it should've completed before)
        #   - and it still can make more steps: 0.1 (continue then)
        tmp_done = self.is_done()
        tmp_max = self.is_max()
        tmp_action_mask = None # TBD later
        tmp_terminate = None # TBD later
        tmp_reward = None # TBD later
        tmp_heuristic_multiplier = 1.0 # helper for reward shaping of partial heuristics
        tmp_repeat_multiplier = 1.0 # helper for reward shaping of coverage-based exploration
        heuristic_list = [
            InvariantHeuristic.no_enum2expr_root(self.curr_trinity_inv),
            InvariantHeuristic.no_duplicate_children(self.curr_trinity_inv)
        ]
        if not all(heuristic_list):
            tmp_heuristic_multiplier = 0.1
        # satisfy all partial heuristics
        if tmp_done:
            self.record_action_seq(self.curr_action_seq)
            tmp_repeat_multiplier = 1.0/InvariantEnvironment.sampled_action_seqs[tuple(self.curr_action_seq)]
            # done, should check the heuristics first
            if not all(heuristic_list):
                # some heuristics won't fit, prevent this invariant from going to checker
                print("# [debug][heuristic][hm={}][rm={:.2f}] seq: {}, inv(before): {}".format(
                    tmp_heuristic_multiplier, tmp_repeat_multiplier, 
                    self.curr_action_seq, self.trinity_inv_to_debugging_inv(self.curr_trinity_inv),
                ))
                tmp_action_mask = [0 for _ in range(len(self.action_list))]
                tmp_terminate = True
                tmp_reward = 1.0 * tmp_heuristic_multiplier * tmp_repeat_multiplier
            else:
                # all good, go to the checker
                print("# [debug][done][hm={}][rm={:.2f}] seq: {}, inv(before): {}".format(
                    tmp_heuristic_multiplier, tmp_repeat_multiplier, 
                    self.curr_action_seq, self.trinity_inv_to_debugging_inv(self.curr_trinity_inv),
                ))
                tmp_verifier_inv = self.trinity_inv_to_verifier_inv(self.curr_trinity_inv)
                tmp_reslist = self.check(self.contract_path, tmp_verifier_inv)
                tmp_action_mask = [0 for _ in range(len(self.action_list))]
                tmp_terminate = True
                if tmp_reslist is None:
                    tmp_reward = 1.0 * tmp_heuristic_multiplier * tmp_repeat_multiplier
                else:
                    if tmp_reslist[0]+tmp_reslist[2]==tmp_reslist[1]+tmp_reslist[3]:
                        # completely correct, remove rm
                        tmp_reward = 10.0*(tmp_reslist[0]+tmp_reslist[2])/(tmp_reslist[1]+tmp_reslist[3])
                    else:
                        # not entirely correct, still need rm
                        tmp_reward = 10.0*(tmp_reslist[0]+tmp_reslist[2])/(tmp_reslist[1]+tmp_reslist[3]) * tmp_heuristic_multiplier * tmp_repeat_multiplier
        else:
            if self.is_max():
                self.record_action_seq(self.curr_action_seq)
                tmp_repeat_multiplier = 1.0/InvariantEnvironment.sampled_action_seqs[tuple(self.curr_action_seq)]
                print("# [debug][max][hm={}][rm={:.2f}] seq: {}, inv: {}".format(
                    tmp_heuristic_multiplier, tmp_repeat_multiplier, 
                    self.curr_action_seq, self.trinity_inv_to_debugging_inv(self.curr_trinity_inv),
                ))
                tmp_action_mask = [0 for _ in range(len(self.action_list))]
                tmp_terminate = True
                tmp_reward = 0.0 * tmp_heuristic_multiplier * tmp_repeat_multiplier
            else:
                # tmp_node here must not be None since it's not done yet
                tmp_node = get_hole_dfs(self.curr_trinity_inv)
                tmp_action_mask = self.get_action_mask(tmp_node.type)
                tmp_terminate = False
                tmp_reward = 0.1 * tmp_heuristic_multiplier * tmp_repeat_multiplier

        return [
            {
                "start": [1],
                "contract": self.contract_observed,
                "action_mask": tmp_action_mask, 
                "nn_seq": self.pad_to_length(self.action_seq_to_nn_seq(self.curr_action_seq), self.max_step),
                "action_seq": self.action_seq_to_nn_seq(list(range(len(self.action_list)))),
            }, 
            tmp_reward, 
            tmp_terminate, 
            {}, # info
        ]

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def render(self, mode="human"):
        pass

    def close(self):
        pass