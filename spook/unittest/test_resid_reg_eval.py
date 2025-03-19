import sys

import numpy as np

sys.path.append("../../")
from spook import SpookL1, SpookL2, SpookPosL1, SpookPosL2

"""
Test the methods that evaluates the residuals and regularization terms
"""

np.random.seed(0)
x_groundtruth = np.random.rand(100)*5 # Na = 100
A = np.random.rand(1000, 100) + 0.02 # 1000 Nshots
Ns = A.shape[0]
b =  A @ x_groundtruth + np.random.randn(Ns)*2e-3
atb = A.T @ b / Ns
ata = A.T @ A / Ns
btb = b.T @ b / Ns

def check_rl2_eval(Solver, ata, atb, btb):
    print("Testing", Solver.__name__, end=' rmse eval...')
    spk = Solver(atb[:,None], ata, mode='contracted',
                 lsparse=1e-20, lsmooth=(1e-20, 1e-20))
    Xo = spk.getXopt()
    # print(abs(Xo-x_groundtruth).mean())
    resid_gt = A @ np.squeeze(Xo) - b
    resid_gt = (np.sum(resid_gt**2)/Ns)**0.5
    print(f"gt-rmse = {resid_gt:.3g}",end='...')
    spk.set_btb(btb)
    rl2_default = spk.residueL2()
    rl2_manual = spk.residueL2(Xo)
    assert np.allclose(rl2_manual, resid_gt), f"{rl2_manual:.2g}!= {resid_gt:.2g}"
    assert np.allclose(rl2_default, resid_gt), f"{rl2_default:.2g}!= {resid_gt:.2g}"
    print(f"Passed on atb.shape={atb.shape}")
    return True

def check_sm_eval(Solver, ata, atb, btb):
    print("Testing", Solver.__name__, end=' reg eval...')
    spk = Solver(atb[:,None], ata, mode='contracted',
                 lsparse=1e-20, lsmooth=(1e-20, 1e-20))
    Xo = spk.getXopt()
    for d in ['w','b']:
        wsm_default = spk.smoothness(dim='w')
        wsm_manual  = spk.smoothness(Xo, dim='w')
        assert np.allclose(wsm_default, wsm_manual), f"{wsm_default:.2g}!= {wsm_manual:.2g}"
    sp_default = spk.sparsity()
    sp_manual  = spk.sparsity(Xo)
    sp_gt = np.sum(Xo**2)**0.5
    if "L1" in Solver.__name__:
        sp_gt = abs(Xo).sum()
    assert np.allclose(sp_gt, sp_default), f"{sp_gt:.2g}!= {sp_default:.2g}"
    assert np.allclose(sp_gt, sp_manual), f"{sp_gt:.2g}!= {sp_manual:.2g}"
    print(f"Passed on atb.shape={atb.shape}")
    return True

for solver in [SpookL2, SpookL1, SpookPosL2, SpookPosL1]:
    check_rl2_eval(solver, ata, atb, btb)

x_groundtruth = np.random.rand(30,20)*5 # Na = 100
A = np.random.rand(2000, 30)+0.01 # 1000 Nshots
Ns = A.shape[0]
b =  A @ x_groundtruth
b += np.random.randn(*(b.shape))*2e-3
atb = A.T @ b / Ns
ata = A.T @ A / Ns
btb = np.trace(b.T @ b) / Ns

for solver in [SpookL2, SpookL1, SpookPosL2, SpookPosL1]:
# for solver in [SpookL2, SpookL1]:
    check_rl2_eval(solver, ata, atb, btb)
    check_sm_eval(solver, ata, atb, btb)
