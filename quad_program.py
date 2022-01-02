#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import scipy.sparse as sps
# from scipy.sparse.linalg import spsolve
import osqp
from .base import SpookBase
from .utils import iso_struct

class SpookQPBase(SpookBase):
    def calc_total_smoother_triu(self):
        """
        CALCULATE Total Smoothness regularization operator
        lsma * (La.T @ La otimes Ib) + lsmb * (Ia otimes Bsmoother)
		
		This method always recalculates

        Returns:
        	The upper triangular part of tensor product matrix, 
        	so DON'T use if the solver is vectorized over dimension b
        """
        Ig = sps.eye(self.Ng)
        temp = sps.kron(self.lsmooth[0] * sps.triu(self._La2), Ig)
        temp += sps.kron(sps.eye(self.Na), self.lsmooth[1] * sps.triu(self._Bsm))
        return temp.tocsc()

    def solve(self, lsparse=None, lsmooth=None):
    	self._updateHyperParams(lsparse, lsmooth)
    	if self.verbose: print("Solving Quad. Prog.")
    	self.res = self._prob.solve()
    	if self.res.info.status != 'solved':
            Warning("Problem not solved. Status: %s"%(self.res.info.status))
        return self.res


class SpookPosL1(SpookQPBase):
	"""
	Positivity + L1 sparsity
	L1 sparsity is just a linear term
	"""
	verbose = False
	def __init__(self, B, A, mode='raw', G=None, lsparse=1, lsmooth=(0.1,0.1), 
        Bsmoother="laplacian"):
		SpookBase.__init__(self, B, A, mode=mode, G=G, 
			lsparse=lsparse, lsmooth=lsmooth, Bsmoother=Bsmoother)
		
		self.__Pcore = sps.triu(self.AGtAG, 0, "csc")
		self._qhalf = - self._Bcontracted.ravel()
		# To compare a new P with the current one, it is more efficient to
        # compare just the upper triangular part, so self._P just caches the
        # upper triangular part
		self._P = self.calcPtriu()
		self.setupProb()

	def calcPtriu(self):
		return self.__Pcore + self.calc_total_smoother_triu()

	def setupProb(self):
		"""
		Create a new OSQP problem
		Upper triangular part of self._P is used in OSQP.setup(), 
		regardless of whether self._P is dense or sparse
		"""
		if self.verbose: print("Setting up the OSQP problem")
		if hasattr(self, "_prob"):
            del self._prob
        P = self._P
        I = sps.eye(P.shape[0], format='csc')
        lb, ub = (np.zeros(P.shape[0]), None)
        self._prob = osqp.OSQP()
        self._prob.setup(P, self._qhalf + 0.5*self.lsparse, I, lb, ub, verbose=False)
        # the factor of 0.5 comes from the convention of OSQP

    def solve(self, lsparse=None, lsmooth=None):
    	if self.verbose: print("Nonnegative constraints and L1 sparsity reg.")
    	return SpookQPBase.solve(self, lsparse, lsmooth)

    def update_lsparse(self, lsparse):
        # Updating lsparse just updates the q parameter in the problem
        self.lsparse = lsparse
        self._prob.update(q = self._qhalf + 0.5*self.lsparse)
        if self.verbose: print("Sparsity hyperparam updated.")

    def update_lsmooth(self, lsmooth):
    	self.lsmooth = lsmooth
    	Pnew = self.calcPtriu()
    	if iso_struct(Pnew, self._P):
    		if self.verbose: print("Structure P matrix remained the same")
    		self._prob.update(Px = Pnew.data)
    		self._P = Pnew
    	else:
    		Warning("Structure of P matrix changed: new non-zero entries emerged. This is a rare situation")
    		self._P = Pnew
    		self.setupProb()

class SpookQP1D(SpookBase):
	def __init__(self):
		pass