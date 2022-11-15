import numpy as np
import scipy.sparse as sps

def adaptive_contraction(A, B, a_slice=None, b_slice=None, keep_dims=False):
    """
    Contract two tensors A, B along axis0, generalizing numpy.tensordot in terms of memory limit.
    Parameters:
        A, B: two tensors stores as arrays/lists/dictionaries. 
              Arrays and lists can be mixed-and-matched, 
              but if one of A,B is a dictionary, the other has to be too, with the same set of keys.
        a_slice, b_slice: optional slicing objects that can be applied to each singleshot data in A, B, respectively
        keep_dims:     controlling flag regarding whether the dimensions of the final tensor is kept. If False, the final tensor will be reshaped into a matrice as if A, B were two matrices with axis0 being the shot id while axis1 being the rest.
    """
    match_keylist, Nshot = check_matchness(A,B)

    # formalizing slice objects
    a_slice = check_sliceobj(a_slice)
    b_slice = check_sliceobj(b_slice)
    
    try:
        Bfull = shots_asarray(B, match_keylist, b_slice)
        Afull = shots_asarray(A, match_keylist, a_slice)
        if not keep_dims:
            Bfull = Bfull.reshape(Bfull.shape[0],-1)
            Afull = Afull.reshape(Afull.shape[0],-1)
        return np.tensordot(Afull, Bfull, axes=(0,0))
    except MemoryError:
    # if True:
        l0 = [0] if match_keylist is None else match_keylist[:1]
        a0 = shots_asarray(A, l0, a_slice)
        b0 = shots_asarray(B, l0, b_slice)
        Na, Nb = a0.size, b0.size
        chunk_size = min(2000, int(2e9/(a0.size+b0.size)))
        print(f"Chunk up by {chunk_size} shots")
        del a0,b0
        segpoints = np.append(np.arange(0,Nshot,chunk_size), Nshot)
        res = 0
        for i, segstart in enumerate(segpoints[:-1]):
            segstop = segpoints[i+1]
            if match_keylist is None:
                shot_select = np.arange(segstart, segstop)
            else:
                shot_select = match_keylist[segstart:segstop]
            Bchunk = shots_asarray(B, shot_select, b_slice)
            Achunk = shots_asarray(A, shot_select, a_slice)
            res += np.tensordot(Achunk, Bchunk, axes=(0,0))
        if not keep_dims:
            res.shape = (Na,Nb)
    return res

def shots_asarray(shots_iterable, shots_select=None, singleshot_slice=None):
    """
    Iterate through an iterable object `shots_iterable`, apply optional `singleshot_slice` to each singleshot array,
    then stack up the sliced arrays into an array, whose axis0 is shot id.
    
    Use `shots_select` to specify which shots to stack into the matrix:
        If the iterable is a dictionary, `shots_select` is a list of keys
        If the iterable is a list of np.ndarray, `shots_select` is a list of indices or a slice
    
    This is a lower-level function that only accepts formated singleshot_slice object, if not None
    """
    if singleshot_slice is None:
        singleshot_slice = slice(None)
    if isinstance(shots_iterable, np.ndarray): # in this case, numpy slicing is just fine 
        out = shots_iterable
        if shots_select is not None:
            out = out[shots_select]
        if isinstance(singleshot_slice, slice):
            out = out[:, singleshot_slice]
        else:
            assert isinstance(singleshot_slice, tuple)
            slc = (slice(None),)+singleshot_slice
            out = out[slc]
        return out
    it0 = 0
    if shots_select is not None and isinstance(shots_iterable, dict): # shots_select is a key list in this case
        it0 = shots_select[0]
    it0 = shots_iterable[it0]
    if sps.issparse(it0): it0 = it0.toarray()
    it0 = it0[singleshot_slice]
    if shots_select is None:
        shots_select = range(len(shots_iterable)) if isinstance(shots_iterable, list) else shots_iterable.keys()
    N1 = it0.shape
    N0 = len(shots_select)
    out = np.empty((N0, *N1), dtype='d')
    
    for j, ky in enumerate(shots_select):
        ar = shots_iterable[ky]
        if isinstance(ar, sps.spmatrix): ar = ar.toarray()
        ar = ar[singleshot_slice]
        out[j,:] = ar[:]
    return out

def allsqsum(A, a_slice=None):
    match_keylist, Nshot = check_matchness(A,A)
    a_slice = check_sliceobj(a_slice)
    try:
        Afull = shots_asarray(A, match_keylist, a_slice)
        return np.sum(abs(Afull)**2)
    except MemoryError:
        l0 = [0] if match_keylist is None else match_keylist[:1]
        a0 = shots_asarray(A, l0, a_slice)
        Na = a0.size
        chunk_size = min(2000, int(2e9/Na))
        # print(f"Chunk up by {chunk_size} shots")
        del a0
        segpoints = np.append(np.arange(0,Nshot,chunk_size), Nshot)
        res = 0
        for i, segstart in enumerate(segpoints[:-1]):
            segstop = segpoints[i+1]
            if match_keylist is None:
                shot_select = np.arange(segstart, segstop)
            else:
                shot_select = match_keylist[segstart:segstop]
            Achunk = shots_asarray(A, shot_select, a_slice)
            res += np.sum(abs(Achunk)**2)
    return res

def adaptive_dot(B, G):
    if isinstance(B, np.ndarray):
        return B @ G
    elif isinstance(B, list):
        return [b @ G for b in B] 
        # b will be automatically densified by @ if it were sparse
    else:
        return {k: b @ G for k,b in B.items()}

def check_matchness(A, B):
    if isinstance(B, dict):
        assert isinstance(A, dict), "B in dict type can only contract with A in dict type"
        lstkeya, lstkeyb = (list(A.keys()), list(B.keys()))
        assert np.setdiff1d(lstkeya, lstkeyb).size == 0, "Keys mismatch."
        lstkeyb.sort()
        return lstkeyb, len(lstkeyb)
    assert len(A)==len(B), f"A, B should have the same dimension at axis0, but len(A)={len(A)}, len(B)={len(B)}"
    return None, len(A)

def check_sliceobj(obj):
    if obj is None:
        return slice(None)
    if isinstance(obj, tuple):
        if len(obj)==2 and (isinstance(obj[0],int) or isinstance(obj[0],float)):
            start, stop = int(min(obj)), int(max(obj))
            return slice(start, stop)
        for ob in obj: 
            assert isinstance(ob, slice), f"Unaccepted slicing spec {obj}"
    return obj