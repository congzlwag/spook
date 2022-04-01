import numpy as np
import scipy.sparse as sps
import sys
sys.path.append("../../")

from spook import SpookBase

np.random.seed(1996)
Ardm = np.random.randn(750,27)
Xtrue = np.random.randn(27,17)
G = np.random.rand(20,17)
B0 = Ardm @ Xtrue
B1 = B0 @ (G.T)

spkraw0 = SpookBase(B0, Ardm)
spkraw1 = SpookBase(B1, Ardm, G=G)

X0 = spkraw0.getXopt()
X1 = spkraw1.getXopt()

