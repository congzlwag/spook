"""
@author: congzlwag
"""
import numpy as np
from .utils import worth_sparsify
from scipy.sparse import coo_matrix
from scipy.sparse.linalg import spsolve
# import osqp

class SpookBase:
    """
    Base class for Spooktroscopy
    Simply pseudoinverse the core problem
    (A otimes G) X = B
    A is specifically photon spectra, shaped (#shot, Nw)
    B is a <=2d data (#shot, Nb)
    G is an optional operator on the dimension Nb
    """
    def __init__(self, B, A, mode="raw", G=None, lsparse=None, lsmooth=None):
        """
        :param mode: "raw" or "contracted"
                     In the "contracted" mode, A is AT@A, B is (AT otimes GT)@B, G is GTG
        """
        if not (mode in ['raw', 'contracted']):
            raise ValueError("Unknown mode: %s. Must be either 'raw' or 'contracted'"%mode)

        # Make sure the class eventually stores AtA, GtG and (At otimes Gt)B
        # All these are presumably dense, especially A, if not, crop in.
        if mode == 'contracted':
            assert isinstance(B, np.ndarray)
            self._AtA = A
            self._Bcontracted = B
            self._GtG = G
            assert A.shape[0] == B.shape[0] and (G is None or B.shape[1]==G.shape[1])
        else:
            B_is_dict = False
            if isinstance(B, np.ndarray):
                assert A.ndim==2 and (B.ndim in [1,2]) and (G is None or G.ndim==2)
                if B.ndim == 1:
                    B = B.reshape((-1,1))
                Ns, Na = A.shape
                Nb = B.shape[1]
            elif isinstance(B, dict):
                assert isinstance(A, dict), "When B is a dict, A has to be a dict too."
                keys = list(A.keys())
                Ns = len(keys)
                Na = A[keys[0]].size
                Nb = B[keys[0]].size
                B_is_dict = True
            else:
                raise TypeError("type(B) can only be either dict or array") 
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
                        GtB[ky] = b @ G
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
        self._Na = self._AtA.shape[0]
        if self._GtG is not None and worth_sparsify(self._GtG):
            self._GtG = coo_matrix(self._GtG)

    def getShape(self):
        ret = {"Na": self._Na, "Ng":self._Bcontracted.shape[1] if self._GtG is None else self._GtG.shape[1]}
        if self._GtG is None:
            ret['Nb'] = ret['Ng']
        return ret

    def solve(self, lsparse=None, lsmooth=None):
        """
        Just for the base class
        Redefine for every derived class
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
        if updated or not hasattr(self,'res'):
            print("Updated")
            self.solve(None, None)
        return self.res.reshape((self._Na, -1))

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


def dict_innerprod(dictA, dictB):
    """
    Inner product of two tensors represented as dictionaries, 
    with the contracted dimension being the keys.
    """
    lsta, keys = (list(dictA.keys()), list(dictB.keys()))
    assert np.setdiff1d(keys, lsta).size == 0, "Keys mismatch."
    keys.sort()

    try:
        B = np.empty((len(keys), dictB[keys[0]].size))
        for j, k in enumerate(keys):
            B[j,:] = dictB[k].flatten()
        A = np.vstack([dictA[k] for k in keys])
        res = A.T @ B
    except MemoryError:
        # print("Chunk accumulating")
        res = 0
        chunk_size = 1000
        key_segments = np.array_split(np.asarray(keys), len(keys)//chunk_size+1)
        key_segs = key_segments if not ('tqdm' in globals()) else tqdm(key_segments)
        for ky_seg in tqdm(key_segs):
            A = np.vstack([dictA[k] for k in (ky_seg)])
            B = np.vstack([dictB[k].flatten() for k in (ky_seg)])
            res += A.T @ B
    return res



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

    shape_dct = spkraw1.getShape()
    Na = shape_dct['Na']
    Ns = Ardm.shape[0]
    Nb = B1.shape[1]
    Ng = shape_dct["Ng"]
    print(Na, Ns, Nb, Ng)
    print(Na*Nb * (Ns+Ng) >= Ns*Ng * (Na+Nb))
