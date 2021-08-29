from abc import ABC, abstractmethod
from typing import Any, List
from ..interpreter import InterpreterError
from ..enumerator import Enumerator
from ..decider import Decider
from ..dsl import Node
from ..logger import get_logger

logger = get_logger('tyrell.synthesizer')


class Synthesizer(ABC):

    _enumerator: Enumerator
    _decider: Decider

    def __init__(self, enumerator: Enumerator, decider: Decider):
        self._enumerator = enumerator
        self._decider = decider

    @property
    def enumerator(self):
        return self._enumerator

    @property
    def decider(self):
        return self._decider

    def synthesize(self):
        '''
        A convenient method to enumerate ASTs until the result passes the analysis.
        Returns the synthesized program, or `None` if the synthesis failed.
        '''
        num_attempts = 0
        fc, prog = self._enumerator.next()
        # print("# fc={}, prog={}".format(fc, prog))
        while prog is not None:
            num_attempts += 1
            logger.debug('Proposed: {}.'.format(prog))
            try:
                # decide whether there's skeleton level deduction
                self._decider.analyze(prog, fc=fc)
                # if you can make here without throwing any exceptions, you are good
                logger.info("Accepted: {}.".format(prog))
                logger.info("Total Attempts: {}.".format(num_attempts))
                return prog
            except InterpreterError as e:
                logger.debug("Rejected: {}.".format(e))
                self._enumerator.update(e)
                fc, prog = self._enumerator.next()
        logger.info('Exhausted.')
        logger.info("Total Attempts: {}".format(num_attempts))
        return None
