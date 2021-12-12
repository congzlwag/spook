"""
@author: congzlwag
"""
import numpy as np
import scipy.sparse as sps
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
    def __init__(self, B, A, mode="raw", G=None):
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
            Ng = 1 if G is None else G.shape[1]
            assert Ns == B.shape[0] and (G is None or Nb == G.shape[0])
            # Which to contract first? Nb or Ns?
            if Na*Nb * (Ns+Ng) >= Ns*Ng * (Na+Nb):
                if G is not None:
                    if B_is_dict:
                        GtB = {}
                        for ky,b in B.items:
                            GtB[ky] = b @ G
                    else:
                        GtB = B @ G
                if B_is_dict:
                    self._Bcontracted = dict_innerprod(A, GtB)
                else:
                    self._Bcontracted = A.T @ GtB
            else:
                if B_is_dict:
                    B = dict_innerprod(A, B)
                else:
                    B = A.T @ B
                if G is not None:
                    B = B @ G
                self._Bcontracted = B
            self._AtA = dict_innerprod(A, A) if B_is_dict else A.T @ A
            self._GtG = None if G is None else G.T @ G

    def getShape(self):
        ret = {"Na": self._AtA.shape[0], "Ng":1 if self._GtG is None else self._GtG.shape[1]}
        if self._GtG is None:
            ret['Nb'] = self._GtG.shape[1]
        return ret

    def solve(self):
        pass

    def getXopt(self):
        pass

def dict_innerprod(dictA, dictB):
    """
    Inner product of two tensors represented as dictionaries, 
    with the contracted dimension being the keys.
    """
    lsta, keys = (list(dictA.keys()), list(dictB.keys()))
    assert np.setdiff1d(keys, lsta).size == 0, "Keys mismatch."
    keys.sorted()
    try:
        B = np.empty((len(keys), dictB[keys[0]].size))
        for j, k in enumerate(keys):
            B[j,:] = dictB[k].flatten()
        A = np.vstack([dictA[k] for k in tqdm(keys)])
        res = A.T @ B
    except MemoryError:
        # print("Chunk accumulating")
        res = 0
        chunk_size = 1000
        key_segments = np.array_split(np.asarray(keys), len(keys)//chunk_size+1)
        for ky_seg in tqdm(key_segments):
            A = np.vstack([dictA[k] for k in (ky_seg)])
            B = np.vstack([dictB[k].flatten() for k in (ky_seg)])
            res += A.T @ B
    return res


def laplacian1D_S(N):
    Lmat = sps.eye(N)*(-2)
    if N > 1:
        b = np.ones(N-1)
        Lmat += sps.diags(b, offsets=1) + sps.diags(b, offsets=-1)
    return Lmat

def iso_struct(csc_mata, csc_matb):
    """Determine whether two csc sparse matrices share the same structure
    """
    if csc_mata.shape != csc_matb.shape:
        return False
    res = (csc_mata.indices == csc_matb.indices).all() 
    res = res and (csc_mata.indptr == csc_matb.indptr).all()
    return res

def normalizedATA(A):
    """
    This will normalize A (not in situ normalization) such that
    sum(A_{ij}^2)/N_A = 1
    i.e. pixel-averaged but shot-accumulated A^2 is 1
    """
    AtA = (A.T) @ A
    scaleA = (np.trace(AtA) / (A.shape[1]))**0.5 # px-wise mean-square
    AtA /= (scaleA**2)
    return AtA, scaleA

def normalizedB(B):
    """
    This will normalize B (not in situ normalization) such that
    sum(B_{ij}^2)/N_B = 1
    i.e. pixel-averaged but shot-accumulated B^2 is 1
    """
    scaleB = np.linalg.norm(B,"fro") / (B.shape[1]**0.5)
    return B/scaleB, scaleB

def comboNormalize(A, B, return_scalefactors=False):
    AtA, scaleA = normalizedATA(A)
    tmp, scaleB = normalizedB(B)
    AtB = (A/scaleA).T @ tmp
    if return_scalefactors:
        return AtA, AtB, scaleA, scaleB
    return AtA, AtB