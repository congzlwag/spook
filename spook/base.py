import numpy as np
import scipy.sparse as sps
from scipy.sparse.linalg import spsolve

# from .utils import dict_allsqsum #dict_innerprod,
from .contraction_utils import adaptive_contraction, adaptive_dot, allsqsum
from .utils import (
    calcL2fromContracted,
    count_delaybin,
    eval_Ng,
    eval_Nw,
    laplacian_square_S,
    worth_sparsify,
)


class SpookBase:
    """
    Base class for Spook solvers
    """
    smoothness_drop_boundaries = True
    verbose = False
    def __init__(self, B, A, mode="raw", G=None, lsparse=None, lsmooth=None,
        Bsmoother="laplacian", Asmoother="laplacian", normalize=True):
        """
        :param mode: "raw" or "contracted" (recommended) or "ADraw"
                     In the "contracted" mode, A is AT@A, B is (AT otimes GT)@B, G is GTG
        :param Bsmoother: the quadratic matrix for smoothness
        :param normalize: If True, then normalize ATA, GTG, obtaining scale factors (sA, sG) and then normalize Bcontracted
                              But it won't be in-place for mode='contracted'
                          If 'inplace', then the normalizations are done in-place
                          If (sA, sG), then cache the scale factors and do nothing on the contracted results
                          If False, do nothing and set (sA, sG) = (1,1)

        """
        assert (mode in ['raw', 'contracted', 'ADraw']), f"Unknown mode: {mode}. Must be either 'raw' or 'contracted' or 'ADraw'"

        # Make sure the class eventually stores AtA, GtG and (At otimes Gt)B
        # All these are presumably dense, especially A, if not, crop in.
        if isinstance(B, np.ndarray) and B.ndim==1:
            B = B[:, None]
        if mode == 'contracted':
            assert isinstance(B, np.ndarray), "B has to be a numpy array in contracted mode"
            assert isinstance(A, np.ndarray), "A has to be a numpy array in contracted mode"
            assert B.ndim in [1,2,3], "B.ndim has to be in [1,2,3]"
            assert A.ndim in [2,4], "A.ndim has to be in [2,4]"
            self.__NaTuple = A.shape[:A.ndim//2]
            Na = np.prod(self.__NaTuple)
            A.shape = (Na,-1)
            B.shape = (Na,-1)
            self._AtA = A
            self._Bcontracted = B
            self._GtG = G
            if G is not None:
                assert (B.shape[-1]==G.shape[-1]), "Shape mismatch: " + \
                "In contracted mode, B & G should have the same dimension at axis=-1, but B.shape={B.shape} while G.shape={G.shape}"
        elif mode == 'raw':
            AtA = adaptive_contraction(A, A, keep_dims=True)
            self.__NaTuple = AtA.shape[:AtA.ndim//2]
            Na = np.prod(self.__NaTuple)
            AtA.shape = (Na,-1)
            # if Na*Nb * (Ns+Ng) >= Ns*Ng * (Na+Nb) and G is not None:
            # contracting BG first is more efficient, but I'll leave it out of this raw mode
            # because an expert should use the contracted mode
            AtB = adaptive_contraction(A, B, keep_dims=False)
            AtB.shape = (Na,-1)
            self._Bcontracted = AtB @ G if G is not None else AtB
            self._TrBtB = allsqsum(B)
            self._AtA = AtA
            self._GtG = None if G is None else G.T @ G
        else: # 'mode'=='ADraw'
            assert isinstance(A, list) or isinstance(A, dict), "A should be a list or dict of (photon_spec, delay_bin_index) or (photon_spec, delay_wavelet)"
            Nt = count_delaybin(A)
            Nw = eval_Nw(A)
            self.__NaTuple = (Nw, Nt)
            self._TrBtB = allsqsum(B)
            BG = B if G is None else adaptive_dot(B, G)
            self._GtG = None if G is None else G.T @ G
            AEtAE = np.zeros((Nw, Nt, Nw, Nt), dtype='d')
            AEtBG = np.zeros((Nw, Nt, eval_Ng(BG)), dtype='d')
            if isinstance(A, list):
                iiter = np.arange(len(A))
            else:
                iiter = list(A.keys())
            for i in iiter:
                ai, ti = A[i]
                bi = BG[i]
                if sps.issparse(bi):
                    bi = bi.toarray().ravel()
                else:
                    bi = np.atleast_1d(bi)
                AEtAE[:, ti, :, ti] += ai[:, None] @ ai[None, :]
                AEtBG[:, ti, :] += ai[:, None] @ bi[None, :]
            self._AtA = AEtAE.reshape(Nw*Nt,-1)
            self._Bcontracted = AEtBG.reshape(Nw*Nt,-1)

        self.lsparse = lsparse
        self.lsmooth = self._parse_lsmooth(lsmooth)
        # self._Na = self._AtA.shape[0]
        if self._GtG is not None and worth_sparsify(self._GtG):
            self._GtG = sps.coo_matrix(self._GtG)
        # self._La2 = laplacian_square_S(self.Na, self.smoothness_drop_boundaries)
        self._Bsm, self._Asm = Bsmoother, Asmoother
        if isinstance(Bsmoother, str) and Bsmoother == "laplacian":
            self._Bsm = laplacian_square_S(self.Ng, self.smoothness_drop_boundaries)
        if isinstance(Asmoother, str) and Asmoother == "laplacian":
            self._Asm = laplacian_square_S(self.NaTuple[0], self.smoothness_drop_boundaries)
        if len(self.NaTuple) > 1:
            assert len(self.NaTuple) == 2
            assert len(self.lsmooth) == 3, "The dataset has a delay axis. Please assign lsmooth=(lsm_w, lsm_g, lsm_t)"
            self._Tsm = laplacian_square_S(self.NaTuple[1], self.smoothness_drop_boundaries)
            self._Tsm = sps.kron(sps.eye(self.NaTuple[0]), self._Tsm)
            self._Asm = sps.kron(self._Asm, sps.eye(self.NaTuple[1]))
        self.normalizeAG(normalize)

    @property
    def Na(self):
        """
        Dimension of single-shot A
        """
        return np.prod(self.__NaTuple)

    @property
    def NaTuple(self):
        """
        Shape of single-shot A.
        Most frequently, it is (Nw,)
        But in the case of delay binning, it is (Nw, Nt)
        """
        return self.__NaTuple

    @property
    def Ng(self):
        """
        (Dimension of X) / Na
        """
        return self._Bcontracted.shape[-1]

    def Asm(self):
        """
        Build the smoothness operator for the dimension of A
        """
        temp = self.lsmooth[0] * self._Asm
        if hasattr(self, "_Tsm"):
            temp += self.lsmooth[2] * self._Tsm
        return temp

    def solve(self, lsparse=None, lsmooth=None):
        """
        ONLY for the base class
        For every derived class, PLEASE REDEFINE!
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

    def getXopt(self, lsparse=None, lsmooth=None, orig_scale=True):
        """
        Solve for the optimal X,
        Parameters
        ------------
        lsparse: float|None,  lsmooth: float|tuple|None
            The regularization hyperparameters
        orig_scale: bool
            whether to scale it back to the original scale
        """
        updated = self._updateHyperParams(lsparse, lsmooth)
        if updated and self.verbose:
            print("Updated")
        if updated or not hasattr(self,'res'):
            self.solve(None, None)
        Xopt_scaled = self.res.reshape((*(self.NaTuple), -1))
        if orig_scale:
            return  Xopt_scaled / self.AGscale
        return Xopt_scaled

    def update_lsparse(self, lsparse):
        """ To be redefined in each derived class
        """
        self.lsparse = lsparse

    def update_lsmooth(self, lsmooth):
        self.lsmooth = lsmooth

    def _updateHyperParams(self, lsparse, lsmooth):
        ret = False
        if lsparse is not None and lsparse != self.lsparse:
            self.update_lsparse(lsparse)
            ret = True
        if lsmooth is not None and self._parse_lsmooth(lsmooth) != self.lsmooth:
            self.update_lsmooth(lsmooth)
            ret = True
        return ret

    def normalizeAG(self, normalize):
        if normalize == False:
            self.__Ascale, self.__Gscale = (1,1)
        elif isinstance(normalize, tuple):
            self.__Ascale, self.__Gscale = normalize
        else: # normalize in ['inplace', True]
            scaleA2 = np.trace(self._AtA) / (self._AtA.shape[1]) # px-wise mean-square
            self.__Ascale = scaleA2**0.5
            if self._GtG is None:
                scaleG2 = 1
            else:
                if isinstance(self._GtG, np.ndarray):
                    scaleG2 = np.trace(self._GtG) / (self.Ng) # px-wise mean-square
                else:
                    scaleG2 = self._GtG.diagonal().mean()
                # Actual normalization happens here
                if normalize == 'inplace':
                    self._GtG /= scaleG2
                else:
                    self._GtG = self._GtG.copy() / scaleG2
            self.__Gscale = scaleG2**0.5
            # Actual normalization happens here
            if normalize == 'inplace':
                self._AtA /= scaleA2
                self._Bcontracted /= self.AGscale
            else:
                self._AtA = self._AtA.copy() / scaleA2
                self._Bcontracted = self._Bcontracted.copy() / self.AGscale


    @property
    def AGscale(self):
        """
        Product s_A * s_G, where s_A and s_G are the normalization factors of A and G.
        """
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

    def residueL2(self, X=None):
        """
        Calculate the rmse of the linear model given variable X, in the original (unscaled) problem
        rmse = |(A otimes G)X - B|_2
        Parameters
        ----------
        X : np.ndarray
            Optimization (unscaled) variable. If None, use the last solved X
        """
        if X is None:
            Xo_scaled = self.getXopt(orig_scale=False)
        else:
            Xo_scaled = X * self.AGscale
        if hasattr(self, "_TrBtB") and self._TrBtB is not None: # Then this is tr(B.T @ B) / scalefactor
            const = self._TrBtB
        else:
            raise ValueError("Please set the constant term in the quadratic form through method:set_btb")
        # Because self._AtA, self._GtG, and self._Bcontracted are all scaled, Xo_scaled is scaled too
        rl2 = calcL2fromContracted(Xo_scaled.reshape((self.Na, -1)),
                                   self._AtA, self._Bcontracted, const, self._GtG)
        # rl2 is the residue L2 norm, of the original (unscaled) problem.
        return rl2

    def set_btb(self, btb: float):
        """
        Set the constant term in the quadratic form
        For 1D problem, btb is the sum of b^2 across all shots
        For 2D problem, btb is the sum of b^2 across all shots and all KE bins
        """
        self._TrBtB = btb

    def accumulate(self, AtA_batch, Bcontracted_batch):
        assert (self.__Ascale, self.__Gscale) == (1,1), "Don't accumulate on normalized spook instance"
        assert AtA_batch.shape == self._AtA.shape
        assert Bcontracted_batch.shape == self._Bcontracted.shape
        self._AtA += AtA_batch
        self._Bcontracted += Bcontracted_batch
        if hasattr(self, "_AGtAG"):
            del self._AGtAG # Clear cache

    def save_prectr(self, inplace=False):
        """
        Export the pre-contracted arrays into a dictionary
        """
        A2, G2, AG = self.__Ascale**2, self.__Gscale**2, self.AGscale
        if inplace:
            ret = {"AtA_nmlz":self._AtA, "GtG_nmlz":self._GtG, "Bcontracted_nmlz":self._Bcontracted,
                   "sA":self.__Ascale, "sG":self.__Gscale}
            Warning("inplace=True saves memory from copying the large arrays by passing out the references."+\
                    "To reuse for instantiating another Spook solver, pass normalize=(sA,sG).")
            return ret
        ret = {}
        ret['AtA'] = self._AtA * A2
        ret['Bcontracted'] = self._Bcontracted * AG
        ret['GtG'] = self._GtG * G2 if self._GtG is not None else None
        return ret

    def _parse_lsmooth(self, lsmooth: [float|tuple|None]):
        if lsmooth is None:
            return (1e-16, 1e-16)
        if isinstance(lsmooth, (int, float)):
            return (lsmooth, 0)
        if isinstance(lsmooth, tuple) and len(lsmooth)==1:
            return (lsmooth[0], 0)
        return lsmooth

    def sparsity(self, X=None):
        """
        L2 norm of the solution. No square, no lsparse*.
        """
        if X is None:
            X = self.getXopt()
        return np.sum(X**2)**0.5

    def smoothness(self, X=None, dim='w'):
        """
        Evaluate the smoothness of variable X, along the dimension dim.
        The smoothness is defined as sqrt(X.T @ LL @ X), where LL is the smoothness operator
            by default LL is the square of finite difference Laplacian operator.
            No lsmooth* is applied.
        Parameters
        ----------
        X: the solution to evaluate
        dim: 'w' or 'b' or 't'
            'w': along the photon frequency axis
            'b': along the interested property axis
            't': along the delay axis (if applicable)
        """
        if X is None:
            X = self.getXopt().reshape((self.Na, -1))
        if dim == 'b':
            sm_l2 = X @ self._Bsm # shape=(Na, Ng)
            return np.sum(X * sm_l2)**0.5

        if dim == 'w':
            ltensor = self._Asm
        elif dim == 't':
            ltensor = self._Tsm
        else:
            raise ValueError("Unknown dimension: "+dim)
        sm_l2 = ltensor @ X # shape=(Na, Ng)
        sm_l2 = np.sum(X * sm_l2)
        return sm_l2**0.5
