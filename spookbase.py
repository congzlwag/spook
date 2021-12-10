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
    B is a generally high dimensional data (#shot, *Nb)
        Nb = (Nb1, ...)
    G is an optional tensor operated on the rest dimensions Nb
    """
    def __init__(self, B, A, mode="raw", G=None):
        """
        :param mode: "raw" or "contracted"
        """
        if not (mode in ['raw', 'contracted']):
            raise ValueError("Unknown mode: %s. Must be either 'raw' or 'contracted'"%mode)

        self.__G = G
        if isinstance(B, np.ndarray):
            if B.ndim == 1:
                B = B.reshape((-1,1))
            if mode == "raw":
                B = np.tensordot(A, B, axes=([0],[0]))
                A = A.T @ A
                
        Ia = sps.eye(Na, dtype=dtype)
        Ib = sps.eye(Nb, dtype=dtype)
        self.__La = sps.kron(laplacian1D_S(Na).astype(dtype), Ib)
        self.__Lb = sps.kron(Ia, laplacian1D_S(Nb).astype(dtype)) if (Nb > 1 and self.lsmooth[1] != 0) else 0
        self.__Pau = sps.kron(A.astype(dtype), Ib)
        self.__q = -2 * B.ravel().astype(dtype)
        self.__Na = Na

    def _getP(self):


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