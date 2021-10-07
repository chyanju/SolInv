import re
import json
import pickle
import subprocess
import numpy as np
from typing import List, Any, Union, Dict

import gym
from gym.utils import seeding

from ..tyrell import spec as S
from ..tyrell import dsl as D
from ..tyrell.interpreter import InvariantInterpreter
from ..tyrell.dsl import Node, HoleNode
from ..tyrell.dsl.utils import derive_dfs, get_hole_dfs

from .invariant_heuristic import InvariantHeuristic
from .error import EnvironmentError

class InvariantEnvironment(gym.Env):
    tspec: S.TyrellSpec
    bulder: D.Builder
    start_type: S.Type
    action_list: List[S.Production]
    max_step: int
    solc_version: str
    interpreter: InvariantInterpreter

    contract_path: str
    contract_json: Any
    contract_tokens: List[Any]
    contract_ids: List[int]
    max_contract_length: int # pad or trim

    stovar_list: List[str]
    stovar_dict: Dict[str, int] # map stovar to id
    stovar_to_spvar: Dict[str, str] # map stovar to <VAR?>
    spvar_to_stovar: Dict[str, str] # map <VAR?> to stovar
    spvar_to_prod: Dict[str, S.Production] # map <VAR?> to Trinity production

    sptok_list: List[str]
    token_list: List[str]
    token_dict: Dict[str, int] # map tokens to id

    curr_inv: Node
    curr_seq: List[int]
    info: Dict[str, Any]

    # note: class static variable
    #       used to track previously sampled sequences, for coverage based exploration
    sampled_sequences = {}

    def __init__(self, config: Dict[str, Any]):
        self.tspec = config["spec"]
        self.builder = D.Builder(self.tspec)
        self.start_type = config["start_type"]
        self.action_list = list(self.tspec.productions())
        self.max_step = config["max_step"]
        self.contract_path = config["contract_path"]
        self.max_contract_length = config["max_contract_length"]
        self.solc_version = config["solc_version"]
        self.interpreter = config["interpreter"]

        # process tokens/vocab
        with open("{}".format(config["token_list_path"]), "rb") as f:
            tmp_token_list = pickle.load(f)
        self.sptok_list = ["<PAD>", "<UNK>", "{", "}", "[", "]", ":", ","]
        # note: token list can contain Trinity productions
        self.token_list = self.sptok_list + self.action_list + tmp_token_list
        self.token_dict = {self.token_list[i]:i for i in range(len(self.token_list))}
        # get stovars list
        self.stovar_list = self.get_contract_stovars(self.contract_path)
        self.stovar_dict = {self.stovar_list[i]:i for i in range(len(self.stovar_list))}
        self.stovar_to_spvar = {self.stovar_list[i]:"<VAR{}>".format(i) for i in range(len(self.stovar_list))}
        self.spvar_to_stovar = {self.stovar_to_spvar[dkey]:dkey for dkey in self.stovar_to_spvar.keys()}
        self.spvar_to_prod = {
            "<VAR{}>".format(i) : self.tspec.get_enum_production_or_raise(self.tspec.get_type("EnumExpr"), "<VAR{}>".format(i))
            for i in range(len(self.stovar_list))
        }

        # tokenize the target contract
        self.contract_json = self.get_contract_ast(self.contract_path, self.solc_version)
        self.contract_tokens = self.rec_tokenize_contract(self.contract_json)
        # pad or trim
        if len(self.contract_tokens) > self.max_contract_length:
            self.contract_tokens = self.contract_tokens[:self.max_contract_length]
        else:
            self.contract_tokens += ["<PAD>" for _ in range(self.max_contract_length-len(self.contract_tokens))]
        self.contract_ids = [self.token_dict[p] for p in self.contract_tokens]

        # initialize internal variables
        _ = self.reset()

        # inherited variables
        self.action_space = gym.spaces.Discrete(len(self.action_list))
        # self.observation_space = Box(-1, len(self.action_list), shape=(self.max_step, ), dtype=np.int32)
        # self.observation_space = Box(0, len(self.action_list)+2, shape=(1, ), dtype=np.int32)
        self.observation_space = gym.spaces.Dict({
            "contract": gym.spaces.Box(0, len(self.token_list), shape=(self.max_contract_length,), dtype=np.int32),
            "action_mask": gym.spaces.Box(0, 1, shape=(len(self.action_list),), dtype=np.int32), # for output layer, no need to + len(sptok_list)
            "inv": gym.spaces.Box(0, len(self.action_list)+len(self.sptok_list), shape=(self.max_step, ), dtype=np.int32), # for encoding layer, need to + len(sptok_list)
        })

    def record_sequence(self, arg_seq):
        # add the sequence to the class level recorder and count
        tup_seq = tuple(arg_seq)
        if tup_seq not in InvariantEnvironment.sampled_sequences.keys():
            InvariantEnvironment.sampled_sequences[tup_seq] = 0
        InvariantEnvironment.sampled_sequences[tup_seq] += 1

    def spinv_to_stoinv(self, arg_str):
        '''
        Maps the spvars back to their corresponding stovars.
        '''
        tmp_str = arg_str
        for dkey in self.spvar_to_stovar.keys():
            tmp_str = tmp_str.replace(dkey, self.spvar_to_stovar[dkey])
        return tmp_str

    def rec_tokenize_contract(self, arg_json):
        '''
        Converts json to a list of tokens (not token ids, yet).
        '''
        if isinstance(arg_json, dict):
            tmp_list = ["{"]
            for dkey in arg_json.keys():
                tmp_list += self.rec_tokenize_contract(dkey)
                tmp_list += [":"]
                tmp_list += self.rec_tokenize_contract(arg_json[dkey])
                tmp_list += [","]
            tmp_list += ["}"]
            return tmp_list
        elif isinstance(arg_json, list):
            tmp_list = ["["]
            for p in arg_json:
                tmp_list += self.rec_tokenize_contract(p)
                tmp_list += [","]
            tmp_list += ["]"]
            return tmp_list
        else:
            tmp_tok = str(arg_json)
            if tmp_tok in self.stovar_dict.keys():
                # it's a stovar, get its corresponding spvar
                return [self.spvar_to_prod[self.stovar_to_spvar[tmp_tok]]]
            elif tmp_tok in self.token_dict.keys():
                return [tmp_tok]
            else:
                return ["<UNK>"]

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

    def get_action_mask(self, arg_type):
        # get action mask that allows for a specific type
        return [1 if arg_type == p.lhs else 0 for p in self.tspec.productions()]

    # fixme: have a better version that considers both the problem and the partial invariant
    def get_curr_state(self):
        # for neural network embedding, i.e., get_curr_state:
        #   - position 0, 1, 2, ..., len(sptok_list) are reserved for special tokens
        #   - so other id = original id + len(sptok_list)
        # for environment representation, i.e., curr_seq:
        #   - special token id = original id - len(sptok_list)
        #   - so other id remian the same
        # pad to max_step
        return [
            self.curr_seq[i]+len(self.sptok_list) if i<len(self.curr_seq) else 0
            for i in range(self.max_step)
        ]

    def is_max(self):
        '''
        Returns whether or not the length of action sequence has already reached the preset limit.
        '''
        # -1 since we always have a <SOS> at the beginning
        return len(self.curr_seq) >= self.max_step

    def is_done(self):
        '''
        Returns whether or not there's still any remaining hole in the current invariant.
        '''
        next_hole = get_hole_dfs(self.curr_inv)
        if next_hole is None:
            return True
        else:
            return False
    
    def reset(self):
        # note: this should return data structure as defined by self.observation_space
        #       not only the state (but also including any action mask)
        self.curr_inv = HoleNode(type=self.start_type)
        # self.curr_seq = [self.token_dict["<SOS>"]-len(self.sptok_list)]
        self.curr_seq = []
        self.done = False
        self.info = {}
        return {
            "contract": self.contract_ids, 
            "action_mask": self.get_action_mask(self.start_type),
            "inv": self.get_curr_state()
        }

    def check(self, arg_contract_path: str, arg_invariant: str):
        ret = subprocess.run(
            "liquidsol-exe {} --task check --check-inv '{}'".format(arg_contract_path, arg_invariant),
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

    def step(self, arg_action_id: int):
        '''
        returns: [state, reward, done, info]
        '''
        if arg_action_id >= len(self.action_list):
            raise EnvironmentError("required: [0, {}), got: {}".format(len(self.action_list), arg_action_id))
        
        if self.is_done():
            raise EnvironmentError("the invariant is already complete; no action is required")

        # perform the action: derive method will raise exceptions by itself
        try:
            sts, new_inv = derive_dfs(self.builder, self.curr_inv, self.action_list[arg_action_id])
        except:
            # Exception: Types don't match, expect Empty, got Expr
            self.curr_seq = self.curr_seq + [arg_action_id]
            tmp_state = self.get_curr_state()
            # you can't fill any hole since the seq terminates with an exception
            tmp_action_mask = [0 for _ in range(len(self.action_list))]
            tmp_reward = 0.0
            tmp_terminate = True
            print("# [debug][done/exception] seq: {}, inv(before): {}".format(self.curr_seq, self.spinv_to_stoinv(str(self.curr_inv))))
            return [
                {
                    "contract": self.contract_ids,
                    "action_mask": tmp_action_mask, 
                    "inv": tmp_state
                }, 
                tmp_reward, 
                tmp_terminate, 
                {}
            ]

        if not sts:
            raise EnvironmentError("node is not expanded, check the implementation")
        # then refresh state
        self.curr_inv = new_inv
        self.curr_seq = self.curr_seq + [arg_action_id]
        tmp_state = self.get_curr_state()
        
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
        tmp_action_mask = None
        tmp_terminate = None
        tmp_reward = None
        tmp_heuristic_multiplier = 1.0 # helper for reward shaping of partial heuristics
        tmp_repeat_multiplier = 1.0 # helper for reward shaping of coverage-based exploration
        heuristic_list = [
            InvariantHeuristic.no_enum2expr_root(self.curr_inv),
            InvariantHeuristic.no_duplicate_children(self.curr_inv)
        ]
        if not all(heuristic_list):
            tmp_heuristic_multiplier = 0.1
        # satisfy all partial heuristics
        if tmp_done:
            self.record_sequence(self.curr_seq)
            tmp_repeat_multiplier = 1.0/InvariantEnvironment.sampled_sequences[tuple(self.curr_seq)]
            # done, should check the heuristics first
            if not all(heuristic_list):
                # some heuristics won't fit, prevent this invariant from going to checker
                print("# [debug][heuristic][hm={}][rm={:.2f}] seq: {}, inv(before): {}".format(
                    tmp_heuristic_multiplier, tmp_repeat_multiplier, self.curr_seq, self.spinv_to_stoinv(str(self.curr_inv)))
                )
                tmp_action_mask = [0 for _ in range(len(self.action_list))]
                tmp_terminate = True
                # tmp_reward = 0.1 # no multiplier
                tmp_reward = 1.0 * tmp_heuristic_multiplier * tmp_repeat_multiplier
            else:
                # all good, go to the checker
                print("# [debug][done][hm={}][rm={:.2f}] seq: {}, inv(before): {}".format(
                    tmp_heuristic_multiplier, tmp_repeat_multiplier, self.curr_seq, self.spinv_to_stoinv(str(self.curr_inv)))
                )
                tmp_strinv0 = self.interpreter.eval(self.curr_inv)
                # note: need to map spvar back to stovars
                tmp_strinv = self.spinv_to_stoinv(tmp_strinv0)
                tmp_reslist = self.check(self.contract_path, tmp_strinv)
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
                self.record_sequence(self.curr_seq)
                tmp_repeat_multiplier = 1.0/InvariantEnvironment.sampled_sequences[tuple(self.curr_seq)]
                print("# [debug][max][hm={}][rm={:.2f}] seq: {}, inv: {}".format(
                    tmp_heuristic_multiplier, tmp_repeat_multiplier, self.curr_seq, self.spinv_to_stoinv(str(self.curr_inv)))
                )
                tmp_action_mask = [0 for _ in range(len(self.action_list))]
                tmp_terminate = True
                tmp_reward = 0.0 * tmp_heuristic_multiplier * tmp_repeat_multiplier
            else:
                # tmp_node here must not be None since it's not done yet
                tmp_node = get_hole_dfs(self.curr_inv)
                tmp_action_mask = self.get_action_mask(tmp_node.type)
                tmp_terminate = False
                tmp_reward = 0.1 * tmp_heuristic_multiplier * tmp_repeat_multiplier


        return [
            {
                "contract": self.contract_ids, 
                "action_mask": tmp_action_mask, 
                "inv": tmp_state
            }, 
            tmp_reward, 
            tmp_terminate, 
            {}
        ]

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def render(self, mode="human"):
        pass

    def close(self):
        pass