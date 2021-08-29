from typing import Callable, NamedTuple, List, Any
from .decider import Decider
from .example_base import Example
from ..interpreter import Interpreter, InterpreterError, EnumAssertion, SkeletonAssertion

class KSExampleAIDecider(Decider):
    _abstract_interpreter: Interpreter
    _examples: List[Example]

    def __init__(self,
                 abstract_interpreter: Interpreter,
                 ks_abstract_interpreter: Interpreter,
                 abstract_examples: List[Example]):
        # note: normal abstract interpreter performs full skeleton level (sketch level) deduction
        #       ks abstract interpreter performs skeleton level (including partial parameters) deduction
        self._abstract_interpreter = abstract_interpreter
        self._ks_abstract_interpreter = ks_abstract_interpreter
        if len(abstract_examples) == 0:
            raise ValueError(
                'ExampleDecider cannot take an empty list of abstract_examples')
        self._abstract_examples = abstract_examples

    @property
    def abstract_interpreter(self):
        return self._abstract_interpreter

    @property
    def ks_abstract_interpreter(self):
        return self._ks_abstract_interpreter

    @property
    def abstract_examples(self):
        return self._abstract_examples

    def analyze(self, prog, **kwargs):
        if "fc" in kwargs and kwargs["fc"]:
            # print("# FC!")
            self.analyze_full_skeleton(prog)
        self.analyze_skeleton(prog)
        # if you make here, you are good to go
        return

    def analyze_skeleton(self, prog):
        '''
        Keystroke setting only, partial parameter level deduction
        '''
        for ex in self._abstract_examples:
            res_expected = ex.output # already abstract
            res_actual = self._ks_abstract_interpreter.eval(prog, ex.input)
            result = self._ks_abstract_interpreter.abs_intersected(res_expected, res_actual)
            if not result:
                # infeasible, throw InterpreterError
                # note: this throws EnumAssertion since it's already partial parameter level deduction
                raise EnumAssertion(
                    context=tuple([None for _ in range(prog._tag["nslot"])]),
                    condition=(lambda comb: comb != self._ks_abstract_interpreter._current_combination),
                    tag="keystroke:skeleton"
                )
        # if you reach here, you are good to go
        return

    def analyze_full_skeleton(self, prog):
        '''
        This calls abstract interpretation to determine whether the current skeleton is feasible.
        Note that this method doesn't have CDCL; it only generalizes to itself (one skeleton, the current one).
        '''
        for ex in self._abstract_examples:
            res_expected = ex.output
            res_actual = self._abstract_interpreter.eval(prog, ex.input, is_concrete=False)
            result = self._abstract_interpreter.abs_intersected(res_expected, res_actual)
            if not result:
                # infeasible, throw InterpreterError
                # fixme: currently the whole speculative output is to blame (guard)
                # fixme: currently the whole program is added to KB, ideally only partial programs are to add
                raise SkeletonAssertion(example=ex, prog=prog, tag=prog._tag["skeleton"], info=None)
        # if you reach here, you are good to go
        # print("# FULL SAYS GOOD")
        return
