import re
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

from .error import EnvironmentError

class InvariantEnvironment(gym.Env):
    tspec: S.TyrellSpec
    bulder: D.Builder
    start_type: S.Type
    action_list: List[S.Production]
    max_step: int
    contract_path: str
    interpreter: InvariantInterpreter

    curr_inv: Node
    curr_seq: List[int]
    info: Dict[str, Any]

    def __init__(self, config: Dict[str, Any]):
        self.tspec = config["spec"]
        self.builder = D.Builder(self.tspec)
        self.start_type = config["start_type"]
        self.action_list = list(self.tspec.productions())
        self.max_step = config["max_step"]
        self.contract_path = config["contract_path"]
        self.interpreter = config["interpreter"]

        # initialize internal variables
        _ = self.reset()

        # inherited variables
        self.action_space = gym.spaces.Discrete(len(self.action_list))
        # self.observation_space = Box(-1, len(self.action_list), shape=(self.max_step, ), dtype=np.int32)
        # self.observation_space = Box(0, len(self.action_list)+2, shape=(1, ), dtype=np.int32)
        self.observation_space = gym.spaces.Dict({
            "action_mask": gym.spaces.Box(0, 1, shape=(len(self.action_list),), dtype=np.int32), # for output layer, no need to +2
            "inv": gym.spaces.Box(0, len(self.action_list)+2, shape=(1, ), dtype=np.int32), # for encoding layer, need to +2
        })

    def get_action_mask(self, arg_type):
        # get action mask that allows for a specific type
        return [1 if arg_type == p.lhs else 0 for p in self.tspec.productions()]

    # fixme: have a better version that considers both the problem and the partial invariant
    def get_curr_state(self):
        # for neural network embedding:
        #   - position 0 is <PAD>
        #   - position 1 is <SOS>
        #   - so other id + 2
        # for environment representation:
        #   - position -2 is <PAD>
        #   - position -1 is <SOS>
        #   - other prod ids are the same with Trinity
        return [self.curr_seq[-1]+2]

    def is_max(self):
        # -1 since we always have a <SOS> at the beginning
        return len(self.curr_seq)-1 >= self.max_step

    def is_done(self):
        next_hole = get_hole_dfs(self.curr_inv)
        if next_hole is None:
            return True
        else:
            return False
    
    def reset(self):
        # note: this should return data structure as defined by self.observation_space
        #       not only the state (but also including any action mask)
        self.curr_inv = HoleNode(type=self.start_type)
        self.curr_seq = [-1]
        self.done = False
        self.info = {}
        return {
            # "action_mask": [1 for _ in range(len(self.action_list))],
            "action_mask": self.get_action_mask(self.start_type),
            "inv": self.get_curr_state()
        }

    def check(self, arg_contract_path: str, arg_invariant: str):
        ret = subprocess.run(
            "liquidsol-exe {} --task check --check-inv '{}'".format(arg_contract_path, arg_invariant),
            shell=True, capture_output=True,
        )
        # print("# [debug] check result: {}".format(ret))
        # print("# [debug] seq: {}, checking {}".format(self.curr_seq, arg_invariant))
        if ret.returncode != 0:
            return None
        ret = ret.stdout.decode("utf-8")
        hard_ok, hard = re.search("Hard: ([0-9]+) / ([0-9]+)", ret).groups()
        soft_ok, soft = re.search("Soft: ([0-9]+) / ([0-9]+)", ret).groups()
        print("# [debug] checking good!!!!!!!!!!!!!!!!!!!!!! result: {}".format([hard_ok, hard, soft_ok, soft]))
        # print("# [debug] result: {}".format(arg_invariant, [hard_ok, hard, soft_ok, soft]))
        return list(map(int, [hard_ok, hard, soft_ok, soft]))

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
            print("# [debug][done/exception] seq: {}, inv(before): {}".format(self.curr_seq, self.curr_inv))
            return [
                {"action_mask": tmp_action_mask, "inv": tmp_state}, 
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
        
        # tmp_action_mask = [1 for _ in range(len(self.action_list))]
        

        # there are different cases
        # if the invariant is complete
        #   - if it passes the checking: +nstep*r
        #   - if it fails the checking: 0.0
        # if the invariant is not complete
        #   - if it can still expand: +1.0
        #   - if it already reaches the max step: -nstep

        tmp_done = self.is_done()
        tmp_max = self.is_max()
        tmp_action_mask = None
        tmp_terminate = None
        tmp_reward = None
        if tmp_done:
            print("# [debug][done] seq: {}, inv(before): {}".format(self.curr_seq, self.curr_inv))
            tmp_strinv = self.interpreter.eval(self.curr_inv)
            tmp_reslist = self.check(self.contract_path, tmp_strinv)
            tmp_action_mask = [0 for _ in range(len(self.action_list))]
            tmp_terminate = True
            if tmp_reslist is None:
                tmp_reward = 0.1
            else:
                tmp_reward = 100.0*(tmp_reslist[0]/tmp_reslist[1]) + 100*(tmp_reslist[2]/tmp_reslist[3])
        else:
            if self.is_max():
                print("# [debug][max] seq: {}, inv: {}".format(self.curr_seq, self.curr_inv))
                tmp_action_mask = [0 for _ in range(len(self.action_list))]
                tmp_terminate = True
                tmp_reward = 0.0
            else:
                # tmp_node here must not be None since it's not done yet
                tmp_node = get_hole_dfs(self.curr_inv)
                tmp_action_mask = self.get_action_mask(tmp_node.type)
                tmp_terminate = False
                tmp_reward = 0.1

        return [
            {"action_mask": tmp_action_mask, "inv": tmp_state}, 
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