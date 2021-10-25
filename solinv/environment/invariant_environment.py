import re
import copy
import json
import pickle
import subprocess
import random
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

from .soltype_ast import get_soltype_ast, soltype_ast_to_igraph

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
    cached_contract_utils = {}

    CONTRACT_MAX_IDS   = 100

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.tspec = config["spec"]
        self.builder = D.Builder(self.tspec)
        self.start_type = config["start_type"]
        self.max_step = config["max_step"]
        self.interpreter = config["interpreter"]

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

        # solType slim AST tokens
        self.special_token_list = ["<PAD>", "<ID>", "<REF>"]
        self.reserved_identifier_token_list = [] # TODO: populate this
        self.reserved_vertex_token_list = sorted([
            '<CONTRACT>',
            # expressions
            '<VAR>',
            'CInt', 'CBool',
            "Unary_not", "!=", ">=", "<=", ">", "<", "==", "||", "&&", 
            "+", "-",  "*", "/", "%", "**",
            "+=", "-=", "*=", "/=",
            'EMapInd', 'EField', 'EHavoc',
            # lvalues
            'LvInd', 'LvFld',
            # statements
            'SAsn', 'SCall', 'SDecl', 'SIf', 'SReturn', 'SWhile', 'SHavoc',
            # declarations
            'DCtor',
            'DFun', 'DFun_arg',
            'DStruct',
            # types
            'TyMapping', 'TyStruct', 'TyArray', 'TyAddress', 'TyInt', 'TyInt256', 'TyBool', 'TyByte'])
        self.reserved_edge_token_list = sorted([
            # expressions
            'Var_name', 'Unary_e', 'Binary_lhs', 'Binary_rhs', 
            'EField_fld', 'EField_struct', 'EMapInd_ind', 'EMapInd_map', 
            # lvalues
            'LvFld_fld', 'LvFld_struct', 'LvInd_ind', 'LvInd_map', 
            # statements
            'SAsn_lhs', 'SAsn_rhs', 'SCall_args', 'SCall_args_first', 'SCall_args_next', 'SCall_name', 
            'SIf_cond', 'SIf_else', 'SIf_then', 'SIf_then_first', 'SIf_then_next', 'SIf_else_first', 'SIf_else_next',
            'SWhile_cond', 'SWhile_body_first', 'SWhile_body_next',
            # declarations
            'DCtor_args', 'DCtor_body', 
            'DFun_args', 'DFun_body', 'DFun_name',
            'DFun_args_first', 'DFun_args_next', 'DFun_arg_name', 'DFun_arg_type',
            'DFun_body_first', 'DFun_body_next', 
            'DVar_expr', 'DVar_name', 'DVar_type', 
            # types
            'TyMapping_dst', 'TyArray_elem',
            # contract
            'DFun_first', 'DFun_next',
            # special
            'contents'])

        # every token in the token list should have a fixed embedding for
        self.base_token_list = self.special_token_list \
                             + self.reserved_identifier_token_list \
                             + self.reserved_vertex_token_list \
                             + self.reserved_edge_token_list
        self.token_list = self.base_token_list + self.fixed_action_list
        self.token_dict = {self.token_list[i]:i for i in range(len(self.token_list))}

        # fixme: here we setup all contracts first to prevent contract id not found error in non local mode
        for i in range(len(config["contracts"])):
            self.setup(config, arg_id=i)

        # this caches contract utils for faster switching
        # between cotnracts in training between different rollouts
        self.curr_contract_id = None # need to reset
        _ = self.reset()

        # inherited variables
        # note: for action space, we are using the maximum available productions
        #       in practice, we may be just using a subset of them, i.e., with some of the flex action not used
        self.action_space = gym.spaces.Discrete(len(self.action_list))
        self.observation_space = gym.spaces.Dict({
            "start": gym.spaces.Box(0,1,shape=(1,),dtype=np.int32),
            "contract_id": gym.spaces.Box(0, InvariantEnvironment.CONTRACT_MAX_IDS, shape=(1,), dtype=np.int32),
            "action_mask": gym.spaces.Box(0, 1, shape=(len(self.action_list),), dtype=np.int32), # for output layer, no need to + len(sptok_list)
            # fixme: 1000 should be MAX_NODES
            "nn_seq": gym.spaces.Box(0, len(self.token_list)+1000, shape=(self.max_step, ), dtype=np.int32), # for encoding layer, need to + len(sptok_list)
            "all_actions": gym.spaces.Box(0, len(self.token_list)+1000, shape=(len(self.action_list),), dtype=np.int32), # for dynamic action output, remain the same for the same contract
        })

    def setup(self, arg_config, arg_id=None):
        if arg_id is None:
            # if no contract is specified, randomly choose one
            self.curr_contract_id = random.choice(list(range(len(arg_config["contracts"]))))
        else:
            self.curr_contract_id = arg_id

        if self.curr_contract_id in InvariantEnvironment.cached_contract_utils.keys():
            # directly pull from cache
            cached = InvariantEnvironment.cached_contract_utils[self.curr_contract_id]
            self.contract_path = cached["contract_path"]
            self.solc_version = cached["solc_version"]
            self.contract_json = cached["contract_json"]
            self.contract_static_env = cached["contract_static_env"]
            self.contract_slim_ast = cached["contract_slim_ast"]
            self.contract_e2n = cached["contract_e2n"]
            self.contract_e2r = cached["contract_e2r"]
            self.contract_igraph = cached["contract_igraph"]
            self.contract_root_id = cached["contract_root_id"]
            self.contract_encoded_igraph = cached["contract_encoded_igraph"]
            self.contract_observed = cached["contract_observed"]
            self.stovar_list = cached["stovar_list"]
            self.stovar_dict = cached["stovar_dict"]
            self.flex_action_to_stovar = cached["flex_action_to_stovar"]
            self.stovar_to_flex_action = cached["stovar_to_flex_action"]
        else:
            # need to start a new process

            self.contract_path = arg_config["contracts"][self.curr_contract_id][0]
            self.solc_version = arg_config["contracts"][self.curr_contract_id][1]

            # ================ #
            # contract related #
            # ================ #
            # tokenize the target contract
            self.contract_json = self.get_contract_ast(self.contract_path, self.solc_version)
            
            self.contract_static_env = {}
            self.contract_slim_ast, vs, es, var_v = get_soltype_ast(self.contract_json)
            self.contract_e2n, self.contract_e2r, self.contract_igraph, self.contract_root_id = soltype_ast_to_igraph(self.contract_slim_ast, vs, es, var_v)
            print(self.contract_e2n)
            print(self.contract_e2r)
            print(self.contract_root_id)
            # self.contract_networkx = igraph.Graph.to_networkx(self.contract_igraph)
            # map tokens to corresponding ids (no variable will show up since the graph is already anonymous)
            self.contract_encoded_igraph = self.contract_igraph.copy()
            for p in self.contract_encoded_igraph.vs:
                try:
                    p["token"] = self.token_dict[p["token"]]
                except KeyError:
                    print(p["token"], self.token_dict.keys())
            for p in self.contract_encoded_igraph.es:
                p["token"] = self.token_dict[p["token"]]
            self.contract_observed = {
                "x": torch.tensor(self.contract_encoded_igraph.vs["token"]).long(),
                "edge_attr": torch.tensor(self.contract_encoded_igraph.es["token"]).long(),
                # shape is: (2, num_edges)
                "edge_index": torch.tensor([
                    [self.contract_encoded_igraph.es[i].source for i in range(len(self.contract_encoded_igraph.es))],
                    [self.contract_encoded_igraph.es[i].target for i in range(len(self.contract_encoded_igraph.es))],
                ]).long()
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

            # store to cache
            InvariantEnvironment.cached_contract_utils[self.curr_contract_id] = {}
            cached = InvariantEnvironment.cached_contract_utils[self.curr_contract_id]
            cached["contract_path"] = self.contract_path
            cached["solc_version"] = self.solc_version
            cached["contract_json"] = self.contract_json
            cached["contract_static_env"] = self.contract_static_env
            cached["contract_slim_ast"] = self.contract_slim_ast
            cached["contract_e2n"] = self.contract_e2n
            cached["contract_e2r"] = self.contract_e2r
            cached["contract_igraph"] = self.contract_igraph
            cached["contract_root_id"] = self.contract_root_id
            cached["contract_encoded_igraph"] = self.contract_encoded_igraph
            cached["contract_observed"] = self.contract_observed
            cached["stovar_list"] = self.stovar_list
            cached["stovar_dict"] = self.stovar_dict
            cached["flex_action_to_stovar"] = self.flex_action_to_stovar
            cached["stovar_to_flex_action"] = self.stovar_to_flex_action

        # ====== #
        # basics #
        # ====== #
        # initialize internal variables
        self.curr_trinity_inv = None # invariant in trinity node structure
        # action_seq: represented using ids from action_list, for internal tracking of the environment
        self.curr_action_seq = None

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
            if p >= len(self.fixed_action_list):
                # flex action
                if self.action_list[p] in self.flex_action_to_stovar.keys():
                    ret_seq.append(
                        self.contract_e2n[self.flex_action_to_stovar[self.action_list[p]]] + len(self.base_token_list)
                    )
                else:
                    # this action does not have corresponding id in graph, use padding
                    ret_seq.append( self.token_dict["<PAD>"] )
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
        if self.curr_contract_id not in InvariantEnvironment.sampled_action_seqs.keys():
            InvariantEnvironment.sampled_action_seqs[self.curr_contract_id] = {}
        if tup_seq not in InvariantEnvironment.sampled_action_seqs[self.curr_contract_id].keys():
            InvariantEnvironment.sampled_action_seqs[self.curr_contract_id][tup_seq] = 0
        InvariantEnvironment.sampled_action_seqs[self.curr_contract_id][tup_seq] += 1

    def get_contract_ast(self, arg_path, arg_solc_version):
        cmd_set = subprocess.run("solc-select use {}".format(arg_solc_version), shell=True, capture_output=True)
        if cmd_set.returncode != 0:
            raise Exception("Error executing solc-select. Check your environment configuration.")

        # Use SolType AST
        cmd = f"liquidsol-exe {arg_path} --task ast --only-last"
        cmd_ast = subprocess.run(cmd, shell=True, capture_output=True)
        if cmd_ast.returncode != 0:
            raise Exception(f"Error executing {cmd}. Check your environment configuration.")
        remove_first_line = lambda s: '\n'.join(s.split('\n')[1:])
        raw_output = cmd_ast.stdout.decode("utf-8")
        raw_json = remove_first_line(raw_output)

        parsed_json = json.loads(raw_json)
        return parsed_json

    def get_contract_stovars(self, arg_path):
        cmd_sto = subprocess.run("liquidsol-exe {} --task vars".format(arg_path), shell=True, capture_output=True)
        assert(cmd_sto.returncode == 0)
        raw_output = cmd_sto.stdout.decode("utf-8")
        lines = raw_output.rstrip().split("\n")
        # split different classes
        break_points = []
        for i in range(len(lines)):
            if lines[i].startswith("Now running"):
                break_points.append(i)
        break_points.append(len(lines))
        # process every block
        tmp_list = []
        for j in range(len(break_points)-1):
            curr_lines = lines[break_points[j]+1:break_points[j+1]]
            assert(len(curr_lines) % 2 == 0)
            n = len(curr_lines) // 2
            for i in range(n):
                tmp_list.append(curr_lines[i])
        # fixme: remove duplicate, this is not super appropriate
        tmp_list = list(set(tmp_list))
        # print("# number of stovars: {}, stovars are: {}".format(len(tmp_list), tmp_list))
        return tmp_list

    def get_action_mask(self, arg_type):
        # get action mask that allows for a specific type
        # also mask out redundant variable productions
        tmp_fixed_mask = [1 if arg_type == p.lhs else 0 for p in self.fixed_action_list]
        tmp_flex_mask = [
            1 if i < len(self.stovar_list) and arg_type == self.flex_action_list[i].lhs else 0 
            for i in range(len(self.flex_action_list))
        ]
        # the order is: action_list = fixed_action_list + flex_action_list
        return tmp_fixed_mask + tmp_flex_mask

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
        self.setup(self.config)
        self.curr_trinity_inv = HoleNode(type=self.start_type)
        self.curr_action_seq = []
        return {
            "start": [1],
            # "contract": self.contract_observed,
            "contract_id": [self.curr_contract_id],
            "action_mask": self.get_action_mask(self.start_type),
            "nn_seq": self.pad_to_length(self.action_seq_to_nn_seq(self.curr_action_seq), self.max_step),
            "all_actions": self.action_seq_to_nn_seq(list(range(len(self.action_list)))),
        }

    def check(self, arg_contract_path: str, arg_verifier_inv: str):
        ret = subprocess.run(
            "liquidsol-exe {} --task check --check-inv '{}' --only-last".format(arg_contract_path, arg_verifier_inv),
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
            # input("Found the ground truth!")
            pass
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
            print("# [debug][done/exception] contract: {}, seq: {}, inv(before): {}".format(
                self.curr_contract_id, self.curr_action_seq, self.trinity_inv_to_debugging_inv(self.curr_trinity_inv),
            ))
            return [
                {
                    # you can't fill any hole since the seq terminates with an exception
                    "start": [1],
                    # "contract": self.contract_observed,
                    "contract_id": [self.curr_contract_id],
                    "action_mask": [0 for _ in range(len(self.action_list))], 
                    "nn_seq": self.pad_to_length(self.action_seq_to_nn_seq(self.curr_action_seq), self.max_step),
                    "all_actions": self.action_seq_to_nn_seq(list(range(len(self.action_list)))),
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
            tmp_repeat_multiplier = 1.0/InvariantEnvironment.sampled_action_seqs[self.curr_contract_id][tuple(self.curr_action_seq)]
            # done, should check the heuristics first
            if not all(heuristic_list):
                # some heuristics won't fit, prevent this invariant from going to checker
                print("# [debug][heuristic][hm={}][rm={:.2f}] contract: {}, seq: {}, inv(before): {}".format(
                    tmp_heuristic_multiplier, tmp_repeat_multiplier, 
                    self.curr_contract_id,
                    self.curr_action_seq, self.trinity_inv_to_debugging_inv(self.curr_trinity_inv),
                ))
                tmp_action_mask = [0 for _ in range(len(self.action_list))]
                tmp_terminate = True
                tmp_reward = 1.0 * tmp_heuristic_multiplier * tmp_repeat_multiplier
            else:
                # all good, go to the checker
                print("# [debug][done][hm={}][rm={:.2f}] contract: {}, seq: {}, inv(before): {}".format(
                    tmp_heuristic_multiplier, tmp_repeat_multiplier, 
                    self.curr_contract_id,
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
                tmp_repeat_multiplier = 1.0/InvariantEnvironment.sampled_action_seqs[self.curr_contract_id][tuple(self.curr_action_seq)]
                print("# [debug][max][hm={}][rm={:.2f}] contract: {}, seq: {}, inv: {}".format(
                    tmp_heuristic_multiplier, tmp_repeat_multiplier, 
                    self.curr_contract_id,
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
                # "contract": self.contract_observed,
                "contract_id": [self.curr_contract_id],
                "action_mask": tmp_action_mask, 
                "nn_seq": self.pad_to_length(self.action_seq_to_nn_seq(self.curr_action_seq), self.max_step),
                "all_actions": self.action_seq_to_nn_seq(list(range(len(self.action_list)))),
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