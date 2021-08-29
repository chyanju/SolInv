import random
import numpy as np
from collections import Counter

from .speculator import Speculator

class KSMorpheusGaussianAbstractSpeculator(Speculator):
    '''
    A conceptual speculator that mimic the prediction by approaching the ground truth using gaussian distribution.
    '''
    def __init__(self, n_speculator, fns_mu, fns_sigma, fn_assemble):
        assert len(fns_mu) == n_speculator, \
            "Number of mu functions should match the number of speculators ({}), got: {}.".format(n_speculator, len(fns_mu))
        assert len(fns_sigma) == n_speculator, \
            "Number of sigma functions should match the number of speculators ({}), got: {}.".format(n_speculator, len(fns_sigma))

        self._n_speculator = n_speculator
        self._fns_mu = fns_mu
        self._fns_sigma = fns_sigma
        self._fn_assemble = fn_assemble

    def speculate(self, ts, masks=None):
        # note: by default, random.gauss returns float; 
        #       abstract values usually need to be converted to int by fn_assemble
        # note: number of masks needs to align with number of speculators
        # note: currently for convenience, a mask is a set of allowed values, 
        #       so that new values can be detected by the "in" operator
        if masks is None:
            tmp_samples = [
                random.gauss( self._fns_mu[i](ts), self._fns_sigma[i](ts) )
                for i in range(self._n_speculator)
            ]
            tmp_abs = self._fn_assemble(tmp_samples)
            return tmp_abs
        else:
            # there are masks, so we should use a faster simulation based on numpy
            assert len(masks) == self._n_speculator, \
                "Number of masks should match the number of speculators, got: {}.".format(masks)

            tmp_samples = []
            for i in range(self._n_speculator):
                # fixme: sampling 100000 times is not an efficient implementation
                tmp0 = np.random.normal( self._fns_mu[i](ts), self._fns_sigma[i](ts), 100000 )
                # note: the floor operation is done beforhand here in preparation of the Counter call
                tmp1 = np.floor(tmp0)
                tmp2 = Counter(tmp1).most_common()
                tmp_target = None
                for p in tmp2:
                    if p[0] in masks[i]:
                        # allowed
                        tmp_target = p[0]
                        break
                if tmp_target is None:
                    assert False, "Mask doesn't overlap with current distribution."
                else:
                    tmp_samples.append(tmp_target)
            tmp_abs = self._fn_assemble(tmp_samples)
            return tmp_abs

