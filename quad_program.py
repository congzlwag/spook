import scipy.sparse as sps
from scipy.sparse.linalg import spsolve
import osqp
from .spookbase import SpookBase

class SpookQP2D(SpookBase):
	def __init__(self, B, A, mode='raw', G=None, ):
		pass


class SpookQP1D(SpookBase):
	def __init__(self):
		pass