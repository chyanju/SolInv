import pandas as pd
import numpy as np

from typing import Tuple, List, Iterator, Any

from .. import spec as S
from ..dsl import Node, AtomNode, ParamNode, ApplyNode, CollapsedNode
from ..visitor import GenericVisitor
from . import Interpreter, PostOrderInterpreter, MorpheusInterpreter, MorpheusAbstractInterpreter, InterpreterError, GeneralError
# from .context import Context
from ..logger import get_logger

logger = get_logger('tyrell.morpheus_partial_interpreter')

class MorpheusPartialInterpreter(PostOrderInterpreter):
    _interpreter: MorpheusInterpreter
    _abstract_interpreter: MorpheusAbstractInterpreter

    def __init__(self,
                 interpreter: MorpheusInterpreter,
                 abstract_interpreter: MorpheusAbstractInterpreter,
                 *args, **kwargs):
        super(MorpheusPartialInterpreter, self).__init__(*args, **kwargs)
        self._interpreter = interpreter
        self._abstract_interpreter = abstract_interpreter

    @property
    def interpreter(self):
        return self._interpreter

    @property
    def abstract_interpreter(self):
        return self._abstract_interpreter

    # hijack the original eval method to perform partial evaluation
    def eval(self, prog: Node, inputs: List[Any], is_concrete=True) -> Any:
        # fixme: what do we do if we have two inputs?
        if is_concrete:
            abstract_inputs = [
                self._abstract_interpreter.assemble_abstract_table(inputs[0], p)
                for p in inputs
            ]
        else:
            abstract_inputs = inputs
        class NodeVisitor(GenericVisitor):
            _interp: MorpheusInterpreter
            _absinterp: MorpheusAbstractInterpreter

            def __init__(self, interp, absinterp):
                self._interp = interp
                self._absinterp = absinterp

            # handled by mixed interpreters
            def visit_collapsed_node(self, collapsed_node: CollapsedNode):
                # concrete interpret the collapsed program and get the output
                tmp_result = self._interp.eval( collapsed_node._node, inputs )
                # need to manually set the intermediate result
                collapsed_node._res = tmp_result
                # turn into abstract form and return
                tmp_abs_res = self._absinterp.assemble_abstract_table( inputs[0], tmp_result )
                return tmp_abs_res

            # handled by abstract interpreter
            def visit_atom_node(self, atom_node: AtomNode):
                method_name = self._eval_method_name(atom_node.type.name)
                method = getattr(self._absinterp, method_name, lambda x: x)
                return method(atom_node.data)

            # handled by abstract interpreter
            def visit_param_node(self, param_node: ParamNode):
                param_index = param_node.index
                if param_index >= len(abstract_inputs):
                    msg = 'Input parameter access({}) out of bound({})'.format(
                        param_index, len(abstract_inputs))
                    raise GeneralError(msg)
                return abstract_inputs[param_index]

            # handled by abstract interpreter
            def visit_apply_node(self, apply_node: ApplyNode):
                in_values = [self.visit(x) for x in apply_node.args]
                method_name = self._eval_method_name(apply_node.name)
                method = getattr(self._absinterp, method_name, self._method_not_found)
                return method(apply_node, in_values)

            def _method_not_found(self, apply_node: ApplyNode, arg_values: List[Any]):
                msg = 'Cannot find required eval method: "{}"'.format(
                    self._eval_method_name(apply_node.name))
                raise NotImplementedError(msg)

            @staticmethod
            def _eval_method_name(name):
                return 'eval_' + name

        node_visitor = NodeVisitor(self._interpreter, self._abstract_interpreter)
        try:
            # try if this node is a root node ("skeleton" field only exists in root node)
            tmp_tag = None
            if isinstance(prog, CollapsedNode):
                tmp_tag = prog._node.tag
            else:
                tmp_tag = prog.tag

            if tmp_tag is not None:
                if "skeleton" in tmp_tag:
                    # yes it's root
                    # then initialize set the _current_combination
                    self._interpreter._current_combination = tuple([None for _ in range(tmp_tag["nslot"])])
            return node_visitor.visit(prog)
        except InterpreterError as e:
            raise