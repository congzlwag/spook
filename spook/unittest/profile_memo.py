import sys
import numpy as np
from matplotlib import pyplot as plt
from memory_profiler import profile

sys.path.append("../../")
from spook import SpookLinSolve


np.random.seed(1996)
Na = 90
Nb = 75
Ns = 1000
Ng = 105
Ardm = np.random.rand(1000, Na)*5
Xtrue = np.random.rand(Na, Nb)
G = np.identity(Ng) - 0.2*np.diag(np.ones(Ng-1),k=-1) - 0.2*np.diag(np.ones(Ng-1),k=1)
G = G[:,:Nb]

B0 = Ardm @ Xtrue
B1 = B0 @ (G.T)
B0 += 1e-3*np.linalg.norm(B0) * np.random.randn(*(B0.shape))
B1 += 1e-3*np.linalg.norm(B1) * np.random.randn(*(B1.shape))

@profile
def main(B, A, G):
	spk1 = SpookLinSolve(B, A, "raw", G, lsparse=1, lsmooth=(0.1,0.1), cache_AGtAG=True)
	X1 = spk1.getXopt(1e-6, (1e-7,1e-8))
	return X1

main(B1, Ardm, G)