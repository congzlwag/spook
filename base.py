"""
@author: congzlwag
"""
import numpy as np
from .utils import worth_sparsify, laplacian_square_S
from .utils import dict_innerprod
import  scipy.sparse as sps
from scipy.sparse.linalg import spsolve

class SpookBase:
    """
    Base class for Spooktroscopy
    Simply pseudoinverse the core problem
    (A otimes G) X = B
    A is specifically photon spectra, shaped (#shot, Nw)
    B is a <=2d data (#shot, Nb)
    G is an optional operator on the dimension Nb
    """
    smoothness_drop_boundaries = True
    def __init__(self, B, A, mode="raw", G=None, lsparse=None, lsmooth=None, 
        Bsmoother="laplacian", pre_normalize=True):
        """
        :param mode: "raw" or "contracted"
                     In the "contracted" mode, A is AT@A, B is (AT otimes GT)@B, G is GTG
        :param Bsmoother: the quadratic matrix for smoothness
        :param pre_normalize: whether or not to normalize ATA, GTG
        """
        if not (mode in ['raw', 'contracted']):
            raise ValueError("Unknown mode: %s. Must be either 'raw' or 'contracted'"%mode)

        # Make sure the class eventually stores AtA, GtG and (At otimes Gt)B
        # All these are presumably dense, especially A, if not, crop in.
        if mode == 'contracted':
            assert isinstance(B, np.ndarray), "B has to be a numpy array in contracted mode"
            assert B.ndim in [1,2], "B.ndim has to be 1 or 2"
            if B.ndim == 1:
                B.reshape((-1,1))
            self._AtA = A
            self._Bcontracted = B
            self._GtG = G
            assert A.shape[0] == B.shape[0] and (G is None or B.shape[1]==G.shape[1])
            self._TrBtB = 0
        else:
            B_is_dict = False
            if isinstance(B, np.ndarray):
                assert A.ndim==2 and (B.ndim in [1,2]) and (G is None or G.ndim==2)
                if B.ndim == 1:
                    B = B.reshape((-1,1))
                Ns, Na = A.shape
                Nb = B.shape[1]
                self._TrBtB = np.trace(B.T @ B)
            elif isinstance(B, dict):
                assert isinstance(A, dict), "When B is a dict, A has to be a dict too."
                keys = list(A.keys())
                Ns = len(keys)
                Na = A[keys[0]].size
                Nb = B[keys[0]].size
                B_is_dict = True
                self._TrBtB = np.trace(dict_innerprod(B, B))
            else:
                raise TypeError("B can only be either a dict or an array") 
            assert Ns == (len(B))
            Ng = Nb if G is None else G.shape[1]
            assert (G is None or Nb == G.shape[0])
            # Simply precontract over Ns When G is None
            # But otherwise one need to ponder on the ordering
            # of Ns vs Nb contraction
            if Na*Nb * (Ns+Ng) >= Ns*Ng * (Na+Nb) and G is not None:
                if B_is_dict:
                    GtB = {}
                    for ky,b in B.items():
                        GtB[ky] = b.ravel() @ G
                else:
                    GtB = B @ G
                B = GtB
            if B_is_dict:
                self._Bcontracted = dict_innerprod(A, B)
            else:
                self._Bcontracted = A.T @ B
            if not (Na*Nb * (Ns+Ng) >= Ns*Ng * (Na+Nb)) and G is not None:
                self._Bcontracted = self._Bcontracted @ G
            self._AtA = dict_innerprod(A, A) if B_is_dict else A.T @ A
            self._GtG = None if G is None else G.T @ G

        self.lsparse = lsparse
        self.lsmooth = lsmooth
        # self._Na = self._AtA.shape[0]
        if self._GtG is not None and worth_sparsify(self._GtG):
            self._GtG = sps.coo_matrix(self._GtG)
        self._La2 = laplacian_square_S(self.Na, self.smoothness_drop_boundaries)
        self._Bsm = Bsmoother
        if isinstance(Bsmoother, str) and Bsmoother == "laplacian":
            self._Bsm = laplacian_square_S(self.Ng, self.smoothness_drop_boundaries)

        self.normalizeAG(pre_normalize)


    @property
    def Na(self):
        return self._AtA.shape[0]

    @property
    def Ng(self):
        if self._GtG is None:
            return self._Bcontracted.shape[1] 
        return self._GtG.shape[1]

    # @property
    # def shape(self):
    #     ret = {"Na": self._Na, "Ng": self._Bcontracted.shape[1] if self._GtG is None else self._GtG.shape[1]}
    #     if self._GtG is None:
    #         ret['Nb'] = ret['Ng']
    #     return ret

    def solve(self, lsparse=None, lsmooth=None):
        """
        Just for the base class
        Please Redefine for every derived class
        """
        self._updateHyperParams(lsparse, lsmooth)
        tmp = np.linalg.solve(self._AtA, self._Bcontracted)
        if self._GtG is None:
            self.res = tmp
        else:
            if isinstance(self._GtG, np.ndarray):
                self.res = np.linalg.solve(self._GtG, tmp.T).T
            else:
                self.res = spsolve(self._GtG.tocsc(), tmp.T).T

    def getXopt(self, lsparse=None, lsmooth=None):
        updated = self._updateHyperParams(lsparse, lsmooth)
        if updated and self.verbose: print("Updated")
        if updated or not hasattr(self,'res'):
            self.solve(None, None)
        return self.res.reshape((self.Na, -1)) / self.AGscale
        # Xo /= (self.__Ascale*self.__Gscale)
        # return Xo

    def update_lsparse(self, lsparse):
        """ To be redefined in each derived class
        """
        self.lsparse = lsparse

    def update_lsmooth(self, lsmooth):
        self.lsmooth = lsmooth

    def _updateHyperParams(self, lsparse, lsmooth):
        # print("updating")
        ret = False
        if lsparse is not None and lsparse != self.lsparse:
            self.update_lsparse(lsparse)
            ret = True
        if lsmooth is not None and lsmooth != self.lsmooth:
            self.update_lsmooth(lsmooth)
            ret = True
        return ret

    def normalizeAG(self, pre_normalize):
        if not pre_normalize:
            self.__Ascale, self.__Gscale = (1,1)
        else:
            scaleA2 = np.trace(self._AtA) / (self._AtA.shape[1]) # px-wise mean-square
            # Actual normalization happens here
            self._AtA /= scaleA2
            self.__Ascale = scaleA2**0.5
            if self._GtG is None:
                scaleG2 = 1
            else:
                if isinstance(self._GtG, np.ndarray):
                    scaleG2 = np.trace(self._GtG) / (self.Ng) # px-wise mean-square
                else:
                    scaleG2 = self._GtG.diagonal().mean()
                # Actual normalization happens here
                self._GtG /= scaleG2
            self.__Gscale = scaleG2**0.5
            # Actual normalization happens here
            self._Bcontracted /= self.AGscale
            # self._TrBtB /= (scaleA2*scaleG2)

    @property
    def AGscale(self):
        return self.__Ascale*self.__Gscale

    @property
    def AGtAG(self):
        """
        Tensor product of AtA and GtG
        Some children classes need this tensor itself, 
        some need its upper triangular part only,
        so I make it a non-cache property
        A child class can cache it/its upper triangular part if necessary
        """
        if hasattr(self, "_AGtAG"):
            return self._AGtAG
        GtG = sps.eye(self.Ng) if self._GtG is None else self._GtG
        if isinstance(GtG, np.ndarray):
            return np.kron(self._AtA, GtG)
        else:
            return sps.kron(self._AtA, GtG)

    def residueL2(self, Tr_BtB=None, normalized=True):
        """
        Calculate the L2 norm of the residue.
        |(A otimes G)X - B|_2
        With A & G normalized
        """
        Xo = self.res.reshape((self.Na, -1))
        quad = Xo.T @ self._AtA @ Xo
        if self._GtG is None:
            quad = np.trace(quad)
        else:
            quad = np.trace(quad @ self._GtG)
        lin = -2 * np.trace(Xo.T @ self._Bcontracted) # This covered the contraction with G
        if hasattr(self, "_TrBtB") and self._TrBtB > 0: # Then this is tr(B.T @ B) / scalefactor
            const = self._TrBtB
        elif Tr_BtB is not None:
            self._TrBtB = Tr_BtB
        else:
            raise ValueError("Please input tr(B.T @ B) through param:Tr_BtB")
        if self.verbose: print("Terms in |residue|_2^2: quad=%.1g, lin=%.1g, const=%.1g"%(quad, lin, const))
        rl2 = (max(quad+lin+const,0))**0.5
        if not normalized: # back to the original scale
            return rl2 * self.AGscale
        return rl2

    def scan_lsparse(self, lsparse_list):
        pass
        
if __name__ == '__main__':
    np.random.seed(1996)
    Ardm = np.random.randn(1000,30)
    Xtrue = np.random.randn(30,10)
    G = np.random.rand(20,10)
    B0 = Ardm @ Xtrue
    B1 = B0 @ (G.T)

    spkraw0 = SpookBase(B0, Ardm)
    spkraw1 = SpookBase(B1, Ardm, G=G)

    X0 = spkraw0.getXopt()
    X1 = spkraw1.getXopt()

    # print(np.allclose(Xtrue, X0), np.allclose(Xtrue, X1))

    # AtA = Ardm.T @ Ardm
    # spkctr0 = SpookBase(Ardm.T @ B0, AtA, "contracted")
    # spkctr1 = SpookBase(Ardm.T @ B1 @ G, AtA, "contracted", G=G.T @ G)
    # X0 = spkctr0.getXopt()
    # X1 = spkctr1.getXopt()
    # print(np.allclose(Xtrue, X0), np.allclose(Xtrue, X1))

    # A_dict = {}
    # B0_dict = {}
    # B1_dict = {}
    # for j, a in enumerate(Ardm):
    #     A_dict[j] = a 
    #     B0_dict[j] = B0[j]
    #     B1_dict[j] = B1[j]

    # spk0 = SpookBase(B0_dict, A_dict, "raw")
    # Xd0 = spk0.getXopt()
    # spk1 = SpookBase(B1_dict, A_dict, "raw", G)
    # Xd1 = spk1.getXopt()

    # print(np.allclose(Xtrue, Xd0), np.allclose(Xtrue, Xd1))

    shape_dct = spkraw1.shape
    Na = shape_dct['Na']
    Ns = Ardm.shape[0]
    Nb = B1.shape[1]
    Ng = shape_dct["Ng"]
    print(Na, Ns, Nb, Ng)
    print(Na*Nb * (Ns+Ng) >= Ns*Ng * (Na+Nb))
