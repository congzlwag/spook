import numpy as np
import scipy.sparse as sps
from matplotlib import pyplot as plt

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

def matricize_tensor_bykey(dct, ky_list, roi=None):
    N1 = np.prod(dct[ky_list[0]].shape) if roi is None else np.ptp(roi)
    ret = np.empty((len(ky_list), N1), dtype='d')
    for j, ky in enumerate(ky_list):
        v = dct[ky]
        if isinstance(v, sps.spmatrix):
            v = v.toarray()
        tmp = v.ravel()
        ret[j, :] = tmp if roi is None else tmp[roi[0]:roi[-1]]
    return ret

def dict_innerprod(dictA, dictB, Aroi=None):
    """
    Inner product of two tensors represented as dictionaries, 
    with the contracted dimension being the keys.
    """
    lsta, keys = (list(dictA.keys()), list(dictB.keys()))
    assert np.setdiff1d(keys, lsta).size == 0, "Keys mismatch."
    keys.sort()

    # ROI
    if Aroi is not None:
        assert dictA[keys[0]].ndim <= 2, "ROI is not supported for ndim>2 data."
        assert len(Aroi)==2 and isinstance(Aroi[0],int) and isinstance(Aroi[-1], int), \
        "Unrecognized ROI for A: %s"%(str(Aroi))
        if Aroi[-1] < Aroi[0]:
            Aroi = np.flip(Aroi)

    try:
        B = matricize_tensor_bykey(dictB, keys)
        A = matricize_tensor_bykey(dictA, keys, Aroi)
        res = A.T @ B
    except MemoryError:
        # print("Chunk accumulating")
        res = 0
        chunk_size = 1000
        key_segments = np.array_split(np.asarray(keys), len(keys)//chunk_size+1)
        key_segs = key_segments if not ('tqdm' in globals()) else tqdm(key_segments)
        for ky_seg in key_segs:
            A = matricize_tensor_bykey(dictA, ky_seg, Aroi)
            B = matricize_tensor_bykey(dictB, ky_seg)
            res += A.T @ B
    return res

def dict_allsqsum(dictB):
    keys = list(dictB.keys())
    try:
        B = matricize_tensor_bykey(dictB, keys)
        res = (B**2).sum()
    except:
        res = 0
        chunk_size = 1000
        key_segments = np.array_split(np.asarray(keys), len(keys)//chunk_size+1)
        key_segs = key_segments if not ('tqdm' in globals()) else tqdm(key_segments)
        for ky_seg in key_segs:
            B = matricize_tensor_bykey(dictB, ky_seg)
            res += (B**2).sum()
    return res


def iso_struct(csc_mata, csc_matb):
    """
    Determine whether two csc sparse matrices share the same structure
    """
    if csc_mata.shape != csc_matb.shape:
        return False
    res = (csc_mata.indices == csc_matb.indices)
    if not isinstance(res, np.ndarray) and res == False:
        return False
    res = res.all() and (csc_mata.indptr == csc_matb.indptr).all()
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

def show_lcurve(log_scan_results, curv_dat, plot):
    """
    Plot the data in a L-curve scan.
    """
    if plot == True:
        # print("Making a new figure")
        fig = plt.figure(figsize=(8,4))
    else:
        fig = plot 
    ax0 = fig.add_subplot(1,2,1)
    sc = ax0.scatter(log_scan_results[:,1],log_scan_results[:,2], c=log_scan_results[:,0])
    cax = fig.colorbar(sc,ax=ax0)
    cax.set_label(r"$\lg \lambda_{sp}$")
    ax0.plot(curv_dat[:,1],curv_dat[:,2],'k')
    ax0.set_xlabel(r"$\lg \|AX-B\|_2$")
    ax0.set_ylabel(r"$\lg h_{sp}(X)$")
    ax2 = fig.add_subplot(2,2,2)
    ax2.plot(curv_dat[:,0],curv_dat[:,3])
    ax2.set_ylabel(r"|Tangent Vec|")
    ax3 = fig.add_subplot(2,2,4)
    ax3.plot(curv_dat[:,0],curv_dat[:,4])
    ax3.set_xlabel(r"$\lg \lambda_{sp}$")
    ax3.set_ylabel(r"Curvature")
    idM = np.argmax(curv_dat[:,-1])
    ax0.plot(curv_dat[idM,1],curv_dat[idM,2], "r+")
    ax3.plot(curv_dat[idM,0],curv_dat[idM,4], "r+")
    fig.tight_layout()
    return fig, idM
