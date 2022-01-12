import scipy.sparse as sps
import numpy as np
from scipy.sparse.linalg import spsolve
from .base import SpookBase
from .utils import laplacian_square_S, worth_sparsify

class SpookLinSolve(SpookBase):
    """
    Spooktroscopy that involves only linear eq solving
    This means:
    no positivity constraint 
    L2 sparsity
    """
    verbose = False
    def __init__(self, B, A, mode="raw", G=None, lsparse=1, lsmooth=(0.1,0.1), 
        Bsmoother="laplacian", **kwargs):
        SpookBase.__init__(self, B, A, mode=mode, G=G, lsparse=lsparse, lsmooth=lsmooth, 
            Bsmoother=Bsmoother, **kwargs)
        # self._Ng = self.shape['Ng']
        # L = laplacian1D_S(self._Na)
        # self._La2 = laplacian_square_S(self._Na, self.smoothness_drop_boundaries)
        # self._Bsm = Bsmoother
        # if isinstance(Bsmoother, str) and Bsmoother == "laplacian":
        #     self._Bsm = laplacian_square_S(self._Ng, self.smoothness_drop_boundaries)
        self.setupProb()

    def setupProb(self):
        need_to_flatten = (self._GtG is not None) or self.lsmooth[1]!=0
        if need_to_flatten:
            self.__setupProbFlat()
        else:
            self.__setupProbVec()

    def __setupProbVec(self):
        # print("Set up a vectorized problem")
        assert self._GtG is None and self.lsmooth[1]==0
        self.qhalf = self._Bcontracted
        self.P = self.lsparse * sps.eye(self.Na) + self.lsmooth[0] * self._La2
        self.P += self._AtA

    def __setupProbFlat(self):
        # print("Set up a flattened problem")
        self.qhalf = self._Bcontracted.ravel()
        self.P = self.lsparse * sps.eye(self.Na) + self.lsmooth[0] * self._La2 
        self.P = sps.kron(self.P, sps.eye(self.Ng))
        if hasattr(self,'_AGtAG'):
            self.P += self._AGtAG
        else:
            self._AGtAG = self.AGtAG # save to avoid recalculating the tensor product
            self.P += self._AGtAG
        self.P += sps.kron(sps.eye(self.Na), self.lsmooth[1]*self._Bsm)

    def update_lsparse(self, lsparse):
        # Updating lsparse won't change need_to_flatten
        self.P += (lsparse - self.lsparse) * sps.eye(self.P.shape[0])
        self.lsparse = lsparse

    def update_lsmooth(self, lsmooth):
        self.lsmooth = lsmooth
        if self._GtG is None and self.lsmooth[1]==0:
            self.__setupProbVec()
        else:
            self.__setupProbFlat()

    def solve(self, lsparse=None, lsmooth=None):
        self._updateHyperParams(lsparse, lsmooth)
        if self.verbose: print("Solving Lin. Eq.")
        if isinstance(self.P, np.ndarray):
            self.res = np.linalg.solve(self.P, self.qhalf)
        else:
            self.res = spsolve(self.P, self.qhalf)

if __name__ == '__main__':
    np.random.seed(1996)
    Na = 7
    Nb = 5
    Ns = 1000
    Ng = 9
    Ardm = np.random.rand(1000, Na)*5
    Xtrue = np.random.rand(Na, Nb)
    G = np.identity(Ng) - 0.2*np.diag(np.ones(Ng-1),k=-1) - 0.2*np.diag(np.ones(Ng-1),k=1)
    G = G[:,:Nb]

    from matplotlib import pyplot as plt
    B0 = Ardm @ Xtrue
    B1 = B0 @ (G.T)
    B0 += 1e-3*np.linalg.norm(B0) * np.random.randn(*(B0.shape))
    B1 += 1e-3*np.linalg.norm(B1) * np.random.randn(*(B1.shape))
    spk0 = SpookLinSolve(B0, Ardm, "raw", lsparse=1, lsmooth=(0.1,0.))
    spk1 = SpookLinSolve(B1, Ardm, "raw", G, lsparse=1, lsmooth=(0.1,0.1))

    # X0 = spk0.getXopt(0, (0, 0))
    X1 = spk1.getXopt(0, (0,0))
    # print(Xtrue)
    # print(X0)
    plt.ion()
    plt.imshow(Xtrue, vmin=0, vmax=1)
    plt.figure()
    plt.imshow(X1, vmin=0, vmax=1)