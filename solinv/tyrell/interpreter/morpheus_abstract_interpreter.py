import pandas as pd
import numpy as np

from collections import namedtuple
from typing import Tuple, List, Iterator, Any

from ..dsl import Node, AtomNode, ParamNode, ApplyNode, CollapsedNode
from ..visitor import GenericVisitor
from .interpreter import Interpreter
from .post_order import PostOrderInterpreter
# from .context import Context
from .error import InterpreterError, GeneralError
from ..spec.interval import *

# re-use the concrete interpreter's abstract domain util functions
# from .morpheus_interpreter import *
from .morpheus_interpreter import get_row, get_col, get_head, get_content

# fixme: currently abstract interpretation rules are hard coded
#        need to dynamically load it from the spec in the next version
class MorpheusAbstractInterpreter(PostOrderInterpreter):
    '''
    A basic abstract interpreter that works on full skeleton level.
    '''

    def __init__(self, *args, **kwargs):
        super(MorpheusAbstractInterpreter, self).__init__(*args, **kwargs)

    def make_abs(self):
        return {
            "row": Interval(IMIN,IMAX),
            "col": Interval(IMIN,IMAX),
            "head": Interval(IMIN,IMAX),
            "content": Interval(IMIN,IMAX),
        }

    def assemble_abstract_table(self, arg_tb0, arg_tb1):
        # arg_tb0 is the base, i.e., one of the exampe input(s)
        tmp_row = get_row(arg_tb1)
        tmp_col = get_col(arg_tb1)
        tmp_head = get_head(arg_tb0, arg_tb1)
        tmp_content = get_content(arg_tb0, arg_tb1)
        return {
            "row": Interval(tmp_row, tmp_row),
            "col": Interval(tmp_col, tmp_col),
            "head": Interval(tmp_head, tmp_head),
            "content": Interval(tmp_content, tmp_content),
        }

    def abs_intersected(self, abs0, abs1):
        # within this framework, abs0 and abs1 have the same key set
        for p in abs0.keys():
            if not interval_is_intersected(abs0[p], abs1[p]):
                return False
        return True

    # hijack eval function: transform the inputs to abstract values before feeding
    def eval(self, prog: Node, inputs: List[Any], is_concrete=True) -> Any:
        # fixme: what do we do if we have two inputs?
        if is_concrete:
            abstract_inputs = [
                self.assemble_abstract_table(inputs[0], p)
                for p in inputs
            ]
        else:
            abstract_inputs = inputs
        class NodeVisitor(GenericVisitor):
            _interp: PostOrderInterpreter

            def __init__(self, interp):
                self._interp = interp

            def visit_atom_node(self, atom_node: AtomNode):
                method_name = self._eval_method_name(atom_node.type.name)
                method = getattr(self._interp, method_name, lambda x: x)
                return method(atom_node.data)

            def visit_param_node(self, param_node: ParamNode):
                param_index = param_node.index
                if param_index >= len(abstract_inputs):
                    msg = 'Input parameter access({}) out of bound({})'.format(
                        param_index, len(abstract_inputs))
                    raise GeneralError(msg)
                return abstract_inputs[param_index]

            def visit_apply_node(self, apply_node: ApplyNode):
                in_values = [self.visit(x) for x in apply_node.args]
                method_name = self._eval_method_name(apply_node.name)
                method = getattr(self._interp, method_name,
                                 self._method_not_found)
                return method(apply_node, in_values)

            def _method_not_found(self, apply_node: ApplyNode, arg_values: List[Any]):
                msg = 'Cannot find required eval method: "{}"'.format(
                    self._eval_method_name(apply_node.name))
                raise NotImplementedError(msg)

            @staticmethod
            def _eval_method_name(name):
                return 'eval_' + name

        node_visitor = NodeVisitor(self)
        try:
            return node_visitor.visit(prog)
        except InterpreterError as e:
            raise

    # ================================== #
    # ======== enum productions ======== #
    # ================================== #

    def eval_ColInt(self, v):
        return None

    def eval_SmallInt(self, v):
        return None

    def eval_ColList(self, v):
        return None

    def eval_AggrFunc(self, v):
        return None

    def eval_NumFunc(self, v):
        return None

    def eval_BoolFunc(self, v):
        return None

    # ====================================== #
    # ======== function productions ======== #
    # ====================================== #

    def eval_select(self, node, args):
        arg_tb, arg_collist = args
        out = self.make_abs()
        out["row"] = interval_binary_op("==", out["row"], arg_tb["row"])
        out["col"] = interval_binary_op("<", out["col"], arg_tb["col"])
        out["head"] = interval_binary_op("<=", out["head"], arg_tb["head"])
        out["content"] = interval_binary_op("<=", out["content"], arg_tb["content"])
        return out

    def eval_unite(self, node, args):
        arg_tb, arg_col0, arg_col1 = args
        out = self.make_abs()
        out["row"] = interval_binary_op("==", out["row"], arg_tb["row"])
        out["col"] = interval_binary_op(
            "==",
            out["col"],
            interval_binary_op("-", arg_tb["col"], Interval(1,1)),
        )
        out["head"] = interval_binary_op(
            "<=",
            out["head"],
            interval_binary_op("+", arg_tb["head"], Interval(1,1)),
        )
        out["content"] = interval_binary_op(
            ">=",
            out["content"],
            interval_binary_op("+", arg_tb["content"], Interval(1,1)),
        )
        return out

    def eval_separate(self, node, args):
        arg_tb, arg_col = args
        out = self.make_abs()
        out["row"] = interval_binary_op("==", out["row"], arg_tb["row"])
        out["col"] = interval_binary_op(
            "==",
            out["col"],
            interval_binary_op("+", arg_tb["col"], Interval(1,1)),
        )
        out["head"] = interval_binary_op(
            "<=",
            out["head"],
            interval_binary_op("+", arg_tb["head"], Interval(2,2)),
        )
        out["content"] = interval_binary_op(
            ">=",
            out["content"],
            interval_binary_op("+", arg_tb["content"], Interval(2,2)),
        )
        return out

    def eval_gather(self, node, args):
        arg_tb, arg_collist = args
        out = self.make_abs()
        out["row"] = interval_binary_op(">=", out["row"], arg_tb["row"])
        out["col"] = interval_binary_op("<=", out["col"], arg_tb["col"])
        out["head"] = interval_binary_op(
            "<=",
            out["head"],
            interval_binary_op("+", arg_tb["head"], Interval(2,2)),
        )
        out["content"] = interval_binary_op(
            "<=",
            out["content"],
            interval_binary_op("+", arg_tb["content"], Interval(2,2)),
        )
        return out

    def eval_spread(self, node, args):
        arg_tb, arg_col0, arg_col1 = args
        out = self.make_abs()
        out["row"] = interval_binary_op("<=", out["row"], arg_tb["row"])
        out["col"] = interval_binary_op(">=", out["col"], arg_tb["col"])
        out["head"] = interval_binary_op("<=", out["head"], arg_tb["content"])
        out["content"] = interval_binary_op("<=", out["content"], arg_tb["content"])
        return out

    def eval_mutate(self, node, args):
        arg_tb, arg_op, arg_col0, arg_col1 = args
        out = self.make_abs()
        out["row"] = interval_binary_op("==", out["row"], arg_tb["row"])
        out["col"] = interval_binary_op(
            "==",
            out["col"],
            interval_binary_op("+", arg_tb["col"], Interval(1,1)),
        )
        out["head"] = interval_binary_op(
            "==",
            out["head"],
            interval_binary_op("+", arg_tb["head"], Interval(1,1)),
        )
        out["content"] = interval_binary_op(">", out["content"], arg_tb["content"])
        out["content"] = interval_binary_op(
            "<=",
            out["content"],
            interval_binary_op("+", arg_tb["content"], arg_tb["row"]),
        )
        return out

    def eval_filter(self, node, args):
        arg_tb, arg_op, arg_col, arg_int = args
        out = self.make_abs()
        out["row"] = interval_binary_op("<", out["row"], arg_tb["row"])
        out["col"] = interval_binary_op("==", out["col"], arg_tb["col"])
        out["head"] = interval_binary_op("==", out["head"], arg_tb["head"])
        out["content"] = interval_binary_op("<=", out["content"], arg_tb["content"])
        return out

    def eval_group(self, node, args):
        arg_tb, arg_col0, arg_op, arg_col1 = args
        out = self.make_abs()
        out["row"] = interval_binary_op("<=", out["row"], arg_tb["row"])
        out["col"] = interval_binary_op(
            "<=",
            out["col"],
            interval_binary_op("+", arg_tb["col"], Interval(1,1)),
        )
        out["head"] = interval_binary_op(">", out["head"], Interval(0,0))
        out["head"] = interval_binary_op(
            "<=",
            out["head"],
            interval_binary_op("+", arg_tb["head"], Interval(1,1)),
        )
        # note: originally it's: out.content <= in.content + in.group + 1
        #       since we don't track group, it becomes: out.content <= in.content + in.row + 1
        out["content"] = interval_binary_op(
            "<=",
            out["content"],
            interval_binary_op(
                "+",
                arg_tb["content"],
                interval_binary_op("+", arg_tb["row"], Interval(1,1)),
            ),
        )
        return out
       










