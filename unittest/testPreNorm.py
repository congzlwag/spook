import sys
import numpy as np
from matplotlib import pyplot as plt

sys.path.append("../../")

from spook import SpookPosL1
from spook.utils import normalizedATA

np.random.seed(1996)
Na = 17
Nb = 9
Ns = 10000
Ng = 11

A = np.random.rand(Ns, Na) * 50
Xtrue = np.zeros((Na, Nb))
bb, aa = np.meshgrid(np.arange(Nb), np.arange(Na))
for p1, p2 in zip([1,-1],[1,-1]):
	tmp = 0.1*(Na+Nb) - abs((aa - Na//2) + p1* (bb - Nb//2) - p2* 0.2*(Na+Nb))
	tmp[tmp<0] = 0
	Xtrue += tmp

plt.ion()
fig = plt.figure()
plt.pcolormesh(Xtrue)
plt.colorbar()

G = np.identity(Ng) - 0.2*np.diag(np.ones(Ng-1),k=-1) - 0.2*np.diag(np.ones(Ng-1),k=1)
G = G[:,:Nb]

B0 = A @ Xtrue
B1 = B0 @ (G.T)
B0 += 1e-3*np.linalg.norm(B0) * np.random.randn(*(B0.shape))
B1 += 1e-3*np.linalg.norm(B1) * np.random.randn(*(B1.shape))

SpookPosL1.verbose = True

spk0 = SpookPosL1(B0, A, "raw", lsparse=1, lsmooth=(0.1,0.01))
AtA, sA = normalizedATA(A)
spk0f= SpookPosL1((A.T @ B0)/sA, AtA, "contracted", lsparse=1, lsmooth=(0.1,0.01), pre_normalize=False)
X0 = spk0.getXopt()
X0f= spk0f.getXopt()/sA

print(np.allclose(X0, X0f))

X0 = spk0.getXopt(0.1, (1e-4,1e-4))
fig = plt.figure()
plt.pcolormesh(X0)
plt.colorbar()
