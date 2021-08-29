from typing import Callable, NamedTuple, List, Any
from .decider import Decider
from .example_base import Example
from ..interpreter import Interpreter, InterpreterError, EnumAssertion, SkeletonAssertion

class ExampleAIDecider(Decider):
    _interpreter: Interpreter
    _abstract_interpreter: Interpreter
    _partial_interpreter: Interpreter
    _examples: List[Example]
    _equal_output: Callable[[Any, Any], bool]

    def __init__(self,
                 interpreter: Interpreter,
                 abstract_interpreter: Interpreter,
                 partial_interpreter: Interpreter,
                 examples: List[Example],
                 equal_output: Callable[[Any, Any], bool] = lambda x, y: x == y):
        self._interpreter = interpreter
        self._abstract_interpreter = abstract_interpreter
        self._partial_interpreter = partial_interpreter
        if len(examples) == 0:
            raise ValueError(
                'ExampleDecider cannot take an empty list of examples')
        self._examples = examples
        self._equal_output = equal_output

    @property
    def interpreter(self):
        return self._interpreter

    @property
    def abstract_interpreter(self):
        return self._abstract_interpreter

    @property
    def partial_interpreter(self):
        return self._partial_interpreter

    @property
    def examples(self):
        return self._examples

    @property
    def equal_output(self):
        return self._equal_output

    def analyze_equality(self, prog):
        for x in self._examples:
            self._equal_output( self.interpreter.eval(prog, x.input), x.output )

    def analyze(self, prog, **kwargs):
        if "fc" in kwargs and kwargs["fc"]:
            # fixme: perform sketch level deduction
            self.analyze_full_skeleton(prog)

        # perform partial evaluation
        self.analyze_partial_evaluation(prog)

        # perform concrete evaluation
        self.analyze_equality(prog)

        # if you make here, you are good to go
        return

    def analyze_full_skeleton(self, prog):
        '''
        This calls abstract interpretation to determine whether the current skeleton is feasible.
        Note that this method doesn't have CDCL; it only generalizes to itself (one skeleton, the current one).
        '''
        for ex in self._examples:
            res_expected = self.abstract_interpreter.assemble_abstract_table(ex.input[0], ex.output)
            res_actual = self.abstract_interpreter.eval(prog, ex.input)
            result = self.abstract_interpreter.abs_intersected(res_expected, res_actual)
            if not result:
                # infeasible, throw InterpreterError
                # fixme: currently the whole speculative output is to blame (guard)
                # fixme: currently the whole program is added to KB, ideally only partial programs are to add
                raise SkeletonAssertion(example=ex, prog=prog, tag=prog._tag["skeleton"], info=None)
        # if you reach here, you are good to go
        return

    def analyze_partial_evaluation(self, prog):
        '''
        This is partial evaluation.
        '''
        # in principle, there can only be at most one CollapsedNode in a program
        # first count the number of CollapsedNode
        ac = prog.__repr__().count("ApplyNode")
        for ex in self._examples:
            for i in range(1, ac):
                # if i==0, then it's sketch level deduction
                # if i==ac, then it's concrete evaluation
                # so here for partial evaluation, we only do [1, ac-1]
                iprog = prog.collapse(i)
                res_expected = self.abstract_interpreter.assemble_abstract_table(ex.input[0], ex.output)
                res_actual = self.partial_interpreter.eval(iprog, ex.input)
                result = self.abstract_interpreter.abs_intersected(res_expected, res_actual)
                if not result:
                    # here what we raise is EnumAssertion
                    raise EnumAssertion(
                        context=self.partial_interpreter.interpreter._current_combination,
                        condition=lambda comb:False,
                        tag="partial",
                    )
        pass