import numpy as np
import scipy.sparse as sps
import sys
sys.path.append("../../")
from spook.utils import dict_innerprod

np.random.seed(10)
BIDs = np.arange(100,150)
A = np.random.randn(BIDs.size,20)
B = np.random.rand(BIDs.size, 8,8)
B[B<0.3] = 0
B[B>0.7] = 2

Adct = {}
Bdct = {}
Bsp_dct = {}
for i, b in enumerate(BIDs):
	Adct[b] = A[i]
	Bdct[b] = B[i]
	Bsp_dct[b] = sps.coo_matrix(B[i])

AtA = A.T @ A
BtB = B.reshape(BIDs.size, -1)

AtB = A.T @ BtB
BtB = BtB.T @ BtB

Aroi = (2,10)
print("Does result from dict_innerprod match with numpy.dot?")
AtBdct = dict_innerprod(Adct, Bdct)
print("\tAtB:", np.allclose(AtB, AtBdct))
AtBdct_roi = dict_innerprod(Adct, Bdct, Aroi)
print("\tAtB in roi:", np.allclose(AtB[Aroi[0]:Aroi[-1]], AtBdct_roi))

AtAdct = dict_innerprod(Adct, Adct)
print("\tAtA:", np.allclose(AtA, AtAdct))
BtBdct = dict_innerprod(Bdct, Bdct)
print("\tBtB:", np.allclose(BtB, BtBdct))
BtBdct = dict_innerprod(Bsp_dct, Bdct)
print("\tB_sp.t @ B:", np.allclose(BtB, BtBdct))
BtBdct = dict_innerprod(Bdct, Bsp_dct)
print("\tB.t @ B_sp:", np.allclose(BtB, BtBdct))