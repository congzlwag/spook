import sys

import numpy as np
from matplotlib import pyplot as plt
from sklearn.model_selection import KFold

sys.path.append("../../")
from spook import SpookL2, XValidation

"""
Test the class that does cross-validation
"""

np.random.seed(0)
x_groundtruth = np.random.rand(100)*5 + 0.05 # Na = 100
A = np.random.rand(2002, 100) + 0.1 # 1000 Nshots
A = A + np.roll(A, 1, axis=1) # make it correlated
Ns = A.shape[0]
b =  A @ x_groundtruth + np.random.randn(Ns)*0.1

def prep_xval_prectr(A, b, k=10):
    """
    Prepare the datasets for cross-validation
    """
    precontracted = []
    Ns = A.shape[0]
    kf = KFold(n_splits=k, shuffle=True)
    for i, (train_index, val_index) in enumerate(kf.split(b)):
        train = pctr(A[train_index], b[train_index])
        val = pctr(A[val_index], b[val_index])
        precontracted.append((train, val))
    return precontracted

def pctr(a,b):
    ns = a.shape[0]
    return {"AtA": a.T @ a /ns,
            "AtB": a.T @ b /ns,
            "BtB": b.T @ b /ns,
            "N": ns}


prectr_train_val = prep_xval_prectr(A, b)

spk_xval = XValidation(SpookL2, prectr_train_val, lsparse=1e-5, lsmooth=1e-5)

# print(spk_xval.calc_residual((1e-5, 1e-5), dset='val', avg=True))
# print(spk_xval.calc_residual((1e-5, 1e-5), dset='val', avg=False))
# print(spk_xval.calc_residual((1e-5, 1e-5), dset='train', avg=True))

np.random.seed(2025)
hyperparams = 10**(np.random.rand(100, 2) * 8 - 9)
train_resid = spk_xval.calc_residual(hyperparams, dset='train', avg=True)
val_resid = spk_xval.calc_residual(hyperparams, dset='val', avg=True)

# print(np.allclose(np.array(train_resid), np.array(val_resid)))

fig, axs = plt.subplots(1, 2)
for ax, resid, title in zip(axs, [train_resid, val_resid], ["Train Residuals", "Validation Residuals"]):
    ax.scatter(hyperparams[:,0], hyperparams[:,1], c=resid, cmap='jet')
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_xlabel("lsparse")
    ax.set_ylabel("lsmooth")
    ax.set_title(title)
    fig.colorbar(ax.collections[0], ax=ax)
plt.tight_layout()
plt.show()
