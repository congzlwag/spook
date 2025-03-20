import numpy as np

from .utils import calcL2fromContracted


class XValidation:
    """
    Class for cross-validation of a solver.
    """
    def __init__(self, solver_class, datasets, **kwargs):
        """
        Parameters
        ----------
        solver_class: a spook solver class
        datasets: list of tuples, each tuple is (train, val),
                  where train and val are dictionaries containing the precontracted data
        kwargs: keyword arguments to pass to the solver class
        """
        self._spksolvers = []
        for train, val in datasets:
            spk = solver_class(train["AtB"], train["AtA"], "contracted", **kwargs)
            if "BtB" not in train:
                raise KeyError("BtB must be provided in the training contraction results for cross-validation")
            btb = parse_btb(train["BtB"])
            spk.set_btb(btb)
            spk.precontracted = {}
            spk.precontracted["train"] = (train["AtA"], parse_atb(train["AtB"]), btb)
            spk.precontracted["val"]   = (val["AtA"], parse_atb(val["AtB"]), parse_btb(val["BtB"]))
            self._spksolvers.append(spk)
        self._spksolvers = tuple(self._spksolvers)

    def fit(self, lsparse=None, lsmooth=None):
        return tuple(spk.getXopt(lsparse, lsmooth)
                     for spk in self._spksolvers)

    def calc_residual(self, hyperparams, dset='val', avg=True):
        """
        Validate the hyperparameters on the chosen set.
        Parameters
        ------
        hyperparams: list of tuples | tuple | 2D array
            The hyperparameters to validate.
            Each tuple must be in the form of (lsparse, lsmooth)
            2D array will be interpreted as a list of (lsparse, *lsmooth)
        dset: str
            which dataset to validate on. Must be either 'train' or 'val'
        avg: bool
            If True, return the average of the residuals.
            Otherwise, return the residuals for each train/val split
        Returns
        -------
        float | np.ndarray
            The L2 norm (no square) of the residuals.
            If avg is True, return the root-mean-squared of the L2 norm over the k splits
        """
        assert dset in ["train", "val"], "dset must be either 'train' or 'val'"
        if isinstance(hyperparams, list) or (isinstance(hyperparams, np.ndarray) and hyperparams.ndim == 2):
            return [self.calc_residual(h, dset, avg) for h in hyperparams]
        if isinstance(hyperparams, tuple):
            assert len(hyperparams) == 2, "Each hyperparam must be in the form of (lsparse, lsmooth)"
        elif isinstance(hyperparams, np.ndarray):
            assert hyperparams.ndim == 1 and hyperparams.size >= 2, "Each hyperparam must be in the form of (lsparse, lsmooth1, lsmooth2, ...)"
            hyperparams = (hyperparams[0], tuple(hyperparams[1:]))
        val_resid = np.zeros(self.k)
        for s, spk in enumerate(self._spksolvers):
            X = spk.getXopt(*hyperparams).reshape((spk.Na, -1))
            pctr = spk.precontracted[dset]
            val_resid[s] = calcL2fromContracted(X, pctr[0], pctr[1], pctr[2], spk._GtG)
        if avg:
            return np.mean(val_resid**2)**0.5 # averaging of L2 norm should be rms
        return val_resid

    @property
    def k(self):
        return len(self._spksolvers)

def parse_btb(btb):
    if np.ndim(btb) == 2: # in case the user forgot to take the trace
        return np.trace(btb)
    if np.ndim(btb) == 1 and btb.size == 1:
        return btb[0]
    if np.ndim(btb) == 0:
        return float(btb)
    raise ValueError("BtB must be a [2D array | 1D array with a single element | float]")

def parse_atb(atb):
    if atb.ndim == 1:
        return atb[:, None]
    return atb
