import numpy as np
import scipy.sparse as sps

def laplacian1D_S(N):
    Lmat = sps.eye(N)*(-2)
    if N > 1:
        b = np.ones(N-1)
        Lmat += sps.diags(b, offsets=1) + sps.diags(b, offsets=-1)
    return Lmat

def worth_sparsify(arr):
	if isinstance(arr, np.ndarray):
		return 3*(arr!=0).sum() < arr.size 
	elif isinstance(arr, sps.spmatrix):
		return 3*arr.nnz < np.prod(arr.shape)

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