import pandas as pd
import numpy as np

from typing import Tuple, List, Iterator, Any

from .. import spec as S
from ..dsl import Node, AtomNode, ParamNode, ApplyNode
from ..visitor import GenericVisitor
from . import Interpreter, PostOrderInterpreter, InterpreterError, GeneralError, EqualityAssertion, ComponentError
from ..logger import get_logger

logger = get_logger('tyrell.invariant_interpreter')

class InvariantInterpreter(PostOrderInterpreter):

    def __init__(self, *args, **kwargs):
        super(InvariantInterpreter, self).__init__(*args, **kwargs)

    def eval(self, prog: Node) -> Any:
        class NodeVisitor(GenericVisitor):
            _interp: PostOrderInterpreter

            def __init__(self, interp):
                self._interp = interp

            def visit_atom_node(self, atom_node: AtomNode):
                method_name = self._eval_method_name(atom_node.type.name)
                method = getattr(self._interp, method_name, lambda x: x)
                return method(atom_node.data)

            def visit_param_node(self, param_node: ParamNode):
                raise NotImplementedError("There's no ParamNode in this interpreter.")

            def visit_apply_node(self, apply_node: ApplyNode):
                in_values = [self.visit(x) for x in apply_node.args]
                method_name = self._eval_method_name(apply_node.name)
                method = getattr(self._interp, method_name,
                                 self._method_not_found)
                return method(apply_node, in_values)

            def _method_not_found(self, apply_node: ApplyNode, arg_values: List[Any]):
                msg = 'Cannot find required eval method: "{}"'.format(self._eval_method_name(apply_node.name))
                raise NotImplementedError(msg)

            @staticmethod
            def _eval_method_name(name):
                return 'eval_' + name

        node_visitor = NodeVisitor(self)
        try:
            return node_visitor.visit(prog)
        except InterpreterError as e:
            raise e

    # ================================== #
    # ======== enum productions ======== #
    # ================================== #
    def eval_EnumExpr(self, v):
        return v

    # ====================================== #
    # ======== function productions ======== #
    # ====================================== #
    def eval_enum2expr(self, node, args):
        arg_enumexpr = args[0]
        return arg_enumexpr

    def eval_and(self, node, args):
        arg_expr0, arg_expr1 = args
        return "({} && {})".format(arg_expr0, arg_expr1)

    def eval_leq(self, node, args):
        arg_expr0, arg_expr1 = args
        return "({} <= {})".format(arg_expr0, arg_expr1)

    def eval_add(self, node, args):
        arg_expr0, arg_expr1 = args
        return "({} + {})".format(arg_expr0, arg_expr1)

    def eval_flatten(self, node, args):
        arg_expr = args[0]
        return "flatten({})".format(arg_expr)

    def eval_sum(self, node, args):
        arg_expr = args[0]
        return "sum({})".format(arg_expr)







