import numpy as np
import scipy.sparse as sps

def laplacian1D_S(N):
    Lmat = sps.eye(N)*(-2)
    if N > 1:
        b = np.ones(N-1)
        Lmat += sps.diags(b, offsets=1) + sps.diags(b, offsets=-1)
    return Lmat

def laplacian_square_S(N, drop_bound):
    L = laplacian1D_S(N)
    if drop_bound:
        if N < 3:
            return 0
        L = (L.tocsc())[1:-1]
    return L.T @ L

def worth_sparsify(arr):
    if isinstance(arr, np.ndarray):
        return 3*(arr!=0).sum() < arr.size 
    elif isinstance(arr, sps.spmatrix):
        return 3*arr.nnz < np.prod(arr.shape)

def dict_innerprod(dictA, dictB, Aroi=None):
    """
    Inner product of two tensors represented as dictionaries, 
    with the contracted dimension being the keys.
    """
    lsta, keys = (list(dictA.keys()), list(dictB.keys()))
    assert np.setdiff1d(keys, lsta).size == 0, "Keys mismatch."
    keys.sort()

    # ROI
    if Aroi is None:
        Aroi = (0, dictA[keys[0]].size)
    else:
        assert len(Aroi)==2 and isinstance(Aroi[0],int) and isinstance(Aroi[-1], int), \
        "Unrecognized ROI for A: %s"%(str(Aroi))
    # if Broi is None:
    #     Broi = (0, dictB[keys[0]].size)
    # else:
    #     assert len(Broi)==2 and isinstance(Broi[0],int) and isinstance(Broi[-1], int), \
    #     "Unrecognized ROI for B: %s"%(str(Broi))

    try:
        B = np.empty((len(keys), np.prod(dictB[keys[0]].shape)))
        for j, k in enumerate(keys):
            b = dictB[k]
            if isinstance(b, sps.spmatrix):
                b = b.toarray()
            B[j,:] = b.ravel()
        A = np.vstack([dictA[k][Aroi[0]:Aroi[-1]] for k in keys])
        res = A.T @ B
    except MemoryError:
        # print("Chunk accumulating")
        res = 0
        chunk_size = 1000
        key_segments = np.array_split(np.asarray(keys), len(keys)//chunk_size+1)
        key_segs = key_segments if not ('tqdm' in globals()) else tqdm(key_segments)
        for ky_seg in key_segs:
            A = np.vstack([dictA[k][Aroi[0]:Aroi[-1]] for k in (ky_seg)])
            B = np.vstack([dictB[k].flatten() for k in (ky_seg)])
            res += A.T @ B
    return res

def iso_struct(csc_mata, csc_matb):
    """
    Determine whether two csc sparse matrices share the same structure
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

