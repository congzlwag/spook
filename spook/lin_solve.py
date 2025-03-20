import numpy as np
import scipy.sparse as sps
from scipy.sparse.linalg import spsolve

from .base import SpookBase

# from memory_profiler import profile

class SpookLinSolve(SpookBase):
    """
    Spooktroscopy that involves only linear eq solving
    This means:
    no positivity constraint
    L2 squared sparsity
    """
    verbose = False
    _cache_AGtAG = False
    # Dominant time complexity comes from linsolve, caching AGtAG is
    # not helpful, so I make it optional.
    def __init__(self, B, A, mode="raw", G=None, lsparse=1, lsmooth=(0.1,0.1),
        **kwargs):
        if 'cache_AGtAG' in kwargs:
            self._cache_AGtAG = kwargs['cache_AGtAG']
            del kwargs['cache_AGtAG']
        SpookBase.__init__(self, B, A, mode=mode, G=G, lsparse=lsparse, lsmooth=lsmooth,
            **kwargs)
        self.setupProb()
        self._spfunc = lambda X: (X**2).sum()

    def setupProb(self):
        need_to_flatten = (self._GtG is not None) or self.Ng==1 or self.lsmooth[1]!=0
        if need_to_flatten:
            # Flattening the (w, e) indices into a 1D index
            self.__setupProbFlat()
        else:
            # Set up the problem into Nb sub-problems
            self.__setupProbVec()

    def __setupProbVec(self):
        # Set up the problem into Nb sub-problems
        if self.verbose:
            print("Set up a vectorized problem")
        assert self._GtG is None and self.lsmooth[1]==0
        self.qhalf = self._Bcontracted
        self.P = self.lsparse * sps.eye(self.Na) + self.Asm()
        self.P += self._AtA

    def __setupProbFlat(self):
        # Set up a single, flattened problem
        self.qhalf = self._Bcontracted.ravel()
        self.P = self.lsparse * sps.eye(self.Na) + self.Asm()
        self.P = sps.kron(self.P, sps.eye(self.Ng))
        tmp = self.AGtAG # The base class' AGtAG first look for attr:_AGtAG
        self.P += tmp    # So _AGtAG will be automatically reused if cached
        if self._cache_AGtAG:
            self._AGtAG = tmp # save to avoid recalculating the tensor product
        else:
            del tmp # release this temporary memo alloc
        self.P += sps.kron(sps.eye(self.Na), self.lsmooth[1]*self._Bsm)
        # Convert to CSC format to support self.P[i,i] += delta_lsparse later
        self.P = sps.csc_matrix(self.P)

    def update_lsparse(self, lsparse):
        # Updating lsparse won't change need_to_flatten
        dlsp = lsparse - self.lsparse
        for i in range(self.P.shape[0]):
            self.P[i,i] += dlsp
        self.lsparse = lsparse

    def update_lsmooth(self, lsmooth):
        self.lsmooth = self._parse_lsmooth(lsmooth)
        need_to_flatten = (self._GtG is not None) or self.Ng==1 or self.lsmooth[1]!=0
        if not need_to_flatten:
            self.__setupProbVec()
        else:
            self.__setupProbFlat()

    def solve(self, lsparse=None, lsmooth=None):
        """
        Solve the linear system, since there is only a Tikhonov regularization but no constraint.
        """
        self._updateHyperParams(lsparse, lsmooth)
        if self.verbose:
            print("Solving Lin. Eq.")
        if isinstance(self.P, np.ndarray):
            self.res = np.linalg.solve(self.P, self.qhalf)
        else:
            self.res = spsolve(self.P, self.qhalf)
