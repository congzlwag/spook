import numpy as np
import scipy.sparse as sps
# from matplotlib import pyplot as plt

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

# def normalizedATA(A):
#     """
#     This will normalize A (not in situ normalization) such that
#     sum(A_{ij}^2)/N_A = 1
#     i.e. pixel-averaged but shot-accumulated A^2 is 1
#     """
#     AtA = (A.T) @ A
#     scaleA = (np.trace(AtA) / (A.shape[1]))**0.5 # px-wise mean-square
#     AtA /= (scaleA**2)
#     return AtA, scaleA

# def normalizedB(B):
#     """
#     This will normalize B (not in situ normalization) such that
#     sum(B_{ij}^2)/N_B = 1
#     i.e. pixel-averaged but shot-accumulated B^2 is 1
#     """
#     scaleB = np.linalg.norm(B,"fro") / (B.shape[1]**0.5)
#     return B/scaleB, scaleB

# def comboNormalize(A, B, return_scalefactors=False):
#     AtA, scaleA = normalizedATA(A)
#     tmp, scaleB = normalizedB(B)
#     AtB = (A/scaleA).T @ tmp
#     if return_scalefactors:
#         return AtA, AtB, scaleA, scaleB
#     return AtA, AtB

def count_delaybin(at_iter):
    at_vals = at_iter.values() if isinstance(at_iter, dict) else at_iter
    tlist = [t for a, t in at_vals]
    return max(tlist)  - min(tlist) + 1

def eval_Nw(at_iter):
    if isinstance(at_iter, list):
        return len(at_iter[0][0])
    at = list(at_iter.values())[0]
    return len(at[0])

def eval_Ng(b_iter):
    if isinstance(b_iter, dict):
        for b0 in b_iter.values():
            if sps.issparse(b0):
                return np.prod(b0.shape)
            return b0.size
    return b_iter[0].size

def calcL2fromContracted(Xo, AtA, Bcontracted, trBtB, GtG=None):
    quad = Xo.T @ AtA @ Xo
    if GtG is None:
        quad = np.trace(quad)
    else:
        quad = np.trace(quad @ GtG)
    lin = -2 * np.trace(Xo.T @ Bcontracted) # This covered the contraction with G
    const = trBtB
    rl2 = (max(quad+lin+const,0))**0.5
    # if not normalized: # back to the original scale
    #     return rl2 * self.AGscale
    return rl2

def scan_lsparse(spk, lsparse_list, calc_curvature=True, plot=False):
    assert hasattr(spk, "_TrBtB") and spk._TrBtB > 0, "To scan l_sparse, make sure spk._TrBtB is cached."
    res = np.zeros((len(lsparse_list),3))
    for ll, lsp in enumerate(lsparse_list):
        spk.solve(lsp, None)
        res[ll,:] = [lsp, spk.residueL2(), spk.sparsity()]
    idc = np.argsort(res[:,0])
    res = res[idc]
    if not calc_curvature:
        return res
    Ninterp_min = 101 # Minimal Number of points in interpolation
    margin = 2  # Number of interpolated points to be ignored at the boundaries during differentiation
    res_alllg = np.log10(res)
    spls = [interp1d(res_alllg[:,0],res_alllg[:,i],"cubic",fill_value="extrapolate") for i in range(1,3)]
    ll = np.linspace(res_alllg[0,0],res_alllg[-1,0],max(2*len(lsparse_list)-1,Ninterp_min))[margin:-margin]
    # Try using spl._spline.derivative
    rr = np.asarray([s(ll) for s in spls])
    tt = np.asarray([(s._spline.derivative(1))(ll) for s in spls])
    qq = np.asarray([(s._spline.derivative(2))(ll) for s in spls])
    kk = np.cross(tt,qq,axisa=0,axisb=0).ravel()
    ss = np.linalg.norm(tt, axis=0).ravel()
    kk /= ss**3 # curvature
    curv_dat = np.vstack((ll,rr,ss,kk)).T
    valid_lam_range = np.arange(ll.size)[ss > 1e-2*ss.max()]
    curv_dat = curv_dat[valid_lam_range[0]:valid_lam_range[-1]+1]
    # print(plot)
    if plot:
        # print("Calling show_lcurve")
        show_lcurve(res_alllg, curv_dat, plot)
    idM = np.argmax(curv_dat[:,-1])
    print(curv_dat[idM,0])
    spk.solve(10**(curv_dat[idM,0]), None)
    return res, curv_dat

def show_lcurve(log_scan_results, curv_dat, fig):
    """
    Plot the data in a L-curve scan.
    """
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

def poisson_nll(pred, data):
    assert pred.shape == data.shape
    msk = data > 0
    pred_msk = pred[msk]
    return -(data[msk] * np.log(pred_msk)).sum() + pred_msk.sum()

def soft_poisson_nll(pred, data, p=0.01):
    if sps.issparse(data):
        assert pred.size == np.prod(data.shape)
        data = data.tocoo().reshape((1,-1))
        msk = data.col[data.data>0]
        data = data.data[data.data>0]
    else:
        assert pred.shape == data.shape
        data = data.ravel()
        msk = data > 0
        data = data[msk]
    pred = pred.ravel()[msk]
    pois_msk = pred > p
    gaus_msk = pred <=p
    ret = np.zeros_like(pred)
    ret[pois_msk] = pred[pois_msk] - data[pois_msk] * np.log(pred[pois_msk])
    x = data[gaus_msk]
    ret[gaus_msk] = ((pred[gaus_msk] - x)**2 - (p-x)**2)/(2*p) + p - x*np.log(p)
    return ret.sum()
