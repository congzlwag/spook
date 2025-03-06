import sys

import numpy as np

sys.path.append("../../")
from spook import SpookLinSolve

np.random.seed(0)
x_groundtruth = np.random.randn(100) # Na = 100
A = np.random.randn(1000, 100) # 1000 Nshots
b =  A @ x_groundtruth
atb = A.T @ b
ata = A.T @ A

def check_result(xo, x_groundtruth):
    if np.allclose(xo, x_groundtruth):
        print("Test Passed")
    else:
        print("Max Abs error", abs(xo-x_groundtruth).max())
        print("Rel error",
              np.linalg.norm(xo-x_groundtruth) / np.linalg.norm(x_groundtruth))
        raise ValueError("Test Failed")

spk = SpookLinSolve(atb[:,None], ata, mode='contracted',
                    lsparse=1, lsmooth=(1e-16, 1e-16))
Xo = spk.getXopt(lsparse=1e-16)
Xo = Xo.ravel()
check_result(Xo, x_groundtruth)

spk = SpookLinSolve(atb[:,None], ata, mode='contracted',
                    lsparse=1, lsmooth=1e-16)
Xo = spk.getXopt(lsparse=1e-16)
Xo = Xo.ravel()
check_result(Xo, x_groundtruth)

Xo = spk.getXopt(lsparse=1e-16, lsmooth=1e-10)
Xo = Xo.ravel()
check_result(Xo, x_groundtruth)
