from abc import ABC, abstractmethod
from typing import Any, List
from ..interpreter import InterpreterError
from ..enumerator import Enumerator
from ..decider import Decider
from ..dsl import Node
from ..logger import get_logger

logger = get_logger('tyrell.ks_synthesizer')


class KSSynthesizer(ABC):
    '''
    Synthesizer based on keystroke setting. It's a coordinator that does not return any solution, but performa the following:
    1. keeps tracks of the knowledge bases (predicates/patterns) in the enumerator, and
    2. listens to caller's termination call and returns the knowledge base
    '''

    _enumerator: Enumerator
    _decider: Decider

    def __init__(self, enumerator: Enumerator, decider: Decider):
        self._enumerator = enumerator
        self._decider = decider

        # checks signal in every time step, if True, stop the run
        self._signal_termination = False

    def set_termination(self):
        self._signal_termination = True

    @property
    def enumerator(self):
        return self._enumerator

    @property
    def decider(self):
        return self._decider

    def run(self, stepwise=False):
        '''
        A convenient method to enumerate ASTs until the result passes the analysis.
        Returns the synthesized program, or `None` if the synthesis failed.

        stepwise: whether or not to propose one program only every time this method is called
        '''
        num_attempts = 0
        fc, prog = self._enumerator.next()
        while prog is not None:

            # forced termination from the caller
            if self._signal_termination:
                logger.info("Signal Termination.")
                return

            num_attempts += 1
            logger.debug('Proposed: {}.'.format(prog))
            # logger.debug('Proposed(fc={}): {}.'.format(fc,prog))
            try:
                # decide whether there's skeleton level deduction
                self._decider.analyze(prog, fc=fc)
                # if you can make here without throwing any exceptions, you are good
                logger.info("Passed: {}.".format(prog))
            except InterpreterError as e:
                logger.debug("Rejected: {}.".format(e))
                self._enumerator.update(e)

            if stepwise:
                return

            fc, prog = self._enumerator.next()

        logger.info('Exhausted.')
        return
