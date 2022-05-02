import sys
import scipy.sparse as sps
import numpy as np
from memory_profiler import profile

@profile
def main(N1, N2):
	P = np.random.randn(N1,N1)
	Ps = sps.coo_matrix(P)
	ret = sps.block_diag([Ps, sps.coo_matrix((N2,N2))])
	del P
	return ret

# scipy.sparse.spmatrix.resize does the expansion in situ
@profile
def main2(N1, N2):
	P = np.random.randn(N1,N1)
	Ps = sps.coo_matrix(P)
	Ps.resize((N1+N2,N1+N2))
	# print((Ps.toarray()[:N1,:N1]==P).all())
	return Ps

# main(1000, 2000)

main2(1000, 2000)