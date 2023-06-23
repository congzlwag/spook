#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import scipy.sparse as sps
import numpy as np
# from scipy.sparse.linalg import spsolve
import osqp
from .base import SpookBase
from .utils import iso_struct

class SpookQPBase(SpookBase):
    verbose = False
    def __init__(self, B, A, mode='raw', G=None, lsparse=1, lsmooth=(0.1,0.1), 
        **kwargs):
        SpookBase.__init__(self, B, A, mode=mode, G=G, 
                lsparse=lsparse, lsmooth=lsmooth, **kwargs)
        Ng = self.Ng
        if not (G is None and lsmooth[1]==0 and Ng>1):
            self._Pcore = sps.triu(self.AGtAG, 0, "csc")
            # self._qhalf = - self._Bcontracted.ravel()
            # To compare a new P with the current one, it is more efficient to
            # compare just the upper triangular part, so self._P just caches the
            # upper triangular part
            self._P = self.calcPtriu()
            # SpookL1 will have self._P in shape (2*Nag, 2*Nag), which has no memory overhead
            # beacuse the other three blocks are all zeros, and resize does it in situ
            self._prob = self.setupProb()
        else:
            self._Pcore = sps.triu(self._AtA, 0, "csc")
            self._P = self.calcPtriu()
            self._probs= [self.setupProb(col) for col in range(Ng)]
        self.set_polish(True)

    def calc_total_smoother_triu(self):
        """
        CALCULATE Total Smoothness regularization operator
        lsma * (La.T @ La otimes Ib) + lsmb * (Ia otimes Bsmoother)
        
        This method always recalculates

        Returns:
            The upper triangular part of tensor product matrix, 
            so DON'T use if the solver is vectorized over dimension b
        """
        Asm = self.Asm()
        Ng_eff = self._Pcore.shape[0] // Asm.shape[0]
        temp = sps.kron(sps.triu(Asm), sps.eye(Ng_eff))
        if Ng_eff > 1:
            temp += sps.kron(sps.eye(self.Na), self.lsmooth[1] * sps.triu(self._Bsm))
        return temp.tocsc()

    def solve(self, lsparse=None, lsmooth=None):
        self._updateHyperParams(lsparse, lsmooth)
        if self.verbose: print("Solving Quad. Prog.")
        self.res = []
        if not hasattr(self,'_probs'):
            solution = self._prob.solve()
            if solution.info.status != 'solved':
                Warning("Problem not solved. Status: %s"%(solution.info.status))
            else:
                self.res = solution.x
        else:
            for col, prob in enumerate(self._probs):
                solution = prob.solve()
                if solution.info.status != 'solved':
                    Warning(f"Problem not solved at col{col}. Status: {solution.info.status}")
                self.res.append(solution.x)
            self.res = np.asarray(self.res).T
        return self.res

    def _update_Pmat(self, Pnew):
        if iso_struct(Pnew, self._P):
            if self.verbose: print("Structure of P matrix remained the same")
            if not hasattr(self,'_probs'):
                self._prob.update(Px = Pnew.data)
            else:
                for prob in self._probs:
                    prob.update(Px = Pnew.data)
            self._P = Pnew
        else:
            Warning("Structure of P matrix changed: new non-zero entries emerged. This is a rare situation")
            self._P = Pnew
            if not hasattr(self,'_probs'):
                self._prob = self.setupProb()
            else:
                self._probs = [self.setupProb(col) for col in range(self.Ng)]
            
    def set_polish(self, polish_bool=True):
        if hasattr(self, "_prob"):
            self._prob.update_settings(polish=polish_bool)
        else:
            for prob in self._probs:
                prob.update_settings(polish=polish_bool)

    def qhalf(self, col=None):
        if col is None:
            return - self._Bcontracted.ravel()
        else:
            return - self._Bcontracted[...,col].ravel()

    def update_lsmooth(self, lsmooth):
        assert len(lsmooth) == len(self.NaTuple)+1
        self.lsmooth = lsmooth
        if hasattr(self, '_probs') and lsmooth[1]!=0:
            if self.verbose: print("Resetting problem to be flattened")
            del self._probs
            self._Pcore = sps.triu(self.AGtAG, 0, "csc")
            self._P = self.calcPtriu()
            self._prob = self.setupProb()
        else:
            Pnew = self.calcPtriu()
            self._update_Pmat(Pnew) 

class SpookPos(SpookQPBase):
    """
    Nonnegativity constraint
    Definitely a Quadratic Program
    """

    def initProb(self):
        """
        Create a new OSQP problem
        Upper triangular part of self._P is used in OSQP.setup(), 
        regardless of whether self._P is dense or sparse
        Child class call this to get bounds and OSQP instance
        """
        if self.verbose: print("Setting up the OSQP problem")
        if hasattr(self, "_prob"):
            del self._prob
        dimNaNg = self._Pcore.shape[0]
        I = sps.eye(dimNaNg, format='csc')
        lb, ub = (np.zeros(dimNaNg), None)
        prob = osqp.OSQP()
        return prob, I, lb, ub

    # def solve(self, lsparse=None, lsmooth=None):
    #     """
    #     This will be overriden by the child classes,
    #     and in a child class' solve method, SpookQPBase.solve should be called
    #     """
    #     raise NotImplementedError("Avoid instantiating SpookPos. Implement solve in a Child Class of SpookPos.")


class SpookPosL1(SpookPos):
    """
    Positivity + L1 sparsity
    L1 sparsity is just a linear term
    """
    # __init__ is directly inherited from SpookPos from SpookQPBase

    def calcPtriu(self):
        return self._Pcore + self.calc_total_smoother_triu()

    def setupProb(self, col=None):
        """
        Create a new OSQP problem
        Upper triangular part of self._P is used in OSQP.setup(), 
        regardless of whether self._P is dense or sparse
        """
        self._spfunc = lambda X: abs(X).sum()
        prob, I, lb, ub = SpookPos.initProb(self)
        P = self._P # calculated in SpookPos.__init__
        prob.setup(P, self.qhalf(col) + 0.5*self.lsparse, I, lb, ub, verbose=False)
        # the factor of 0.5 comes from the convention of OSQP
        return prob

    def solve(self, lsparse=None, lsmooth=None):
        if self.verbose: print("Nonnegative constraints and L1 sparsity reg.")
        return SpookQPBase.solve(self, lsparse, lsmooth)

    def update_lsparse(self, lsparse):
        # Updating lsparse just updates the q parameter in the problem
        self.lsparse = lsparse
        if not hasattr(self,'_probs'):
            qhalf = - self._Bcontracted.ravel()
            self._prob.update(q = qhalf + 0.5*self.lsparse)
        else:
            for col, prob in enumerate(self._probs):
                qhalf = - self._Bcontracted[...,col].ravel()
                prob.update(q = qhalf + 0.5*self.lsparse)
        if self.verbose: print("Sparsity hyperparam updated.")

    # def update_lsmooth(self, lsmooth):
    #     self.lsmooth = lsmooth
    #     Pnew = self.calcPtriu()
    #     self._update_Pmat(Pnew)

    # def sparsity(self, X=None):
    #     if X is None:
    #         X = self.res
    #     X = X.ravel()
    #     return abs(X).sum()

class SpookPosL2(SpookPos):
    """
    Non-negativity + L2^2 sparsity, i.e. Ridge w/ Non-negativity constraints
    L1 sparsity is just a linear term
    """
    # __init__ is directly inherited from SpookPos from SpookQPBase
    def calcPtriu(self):
        retP = self._Pcore + self.calc_total_smoother_triu()
        retP += (sps.eye(self._Pcore.shape[0])*self.lsparse).tocsc()
        return retP

    def setupProb(self, col=None):
        """
        Create a new OSQP problem
        Upper triangular part of self._P is used in OSQP.setup(), 
        regardless of whether self._P is dense or sparse
        """
        self._spfunc = lambda X: (X**2).sum()
        prob, I, lb, ub = SpookPos.initProb(self)
        prob.setup(self._P, self.qhalf(col), I, lb, ub, verbose=False)
        # the factor of 0.5 comes from the convention of OSQP
        return prob

    def solve(self, lsparse=None, lsmooth=None):
        if self.verbose: print("Nonnegative constraints and L2 sparsity reg.")
        return SpookQPBase.solve(self, lsparse, lsmooth)

    def update_lsparse(self, lsparse):
        # Updating lsparse involves updating P
        Pnew = self._P + (sps.eye(self._Pcore.shape[0])*(lsparse-self.lsparse)).tocsc()
        self.lsparse = lsparse
        self._update_Pmat(Pnew)

    # def update_lsmooth(self, lsmooth):
    #     self.lsmooth = lsmooth
    #     Pnew = self.calcPtriu()
    #     self._update_Pmat(Pnew)

class SpookL1(SpookQPBase):
    """
    L1 sparsity w/o non-negativity. 
    This is lasso, which requires auxilary variables
    """
    def calcPtriu(self):
        retP = self._Pcore + self.calc_total_smoother_triu()
        Nag  = self._Pcore.shape[0]
        retP.resize((2*Nag, 2*Nag))
        return retP

    def setupProb(self,col=None):
        """
        Create a new OSQP problem
        Upper triangular part of self._P is used in OSQP.setup(), 
        regardless of whether self._P is dense or sparse
        """
        self._spfunc = lambda X: abs(X).sum()
        if self.verbose: print("Setting up the OSQP problem")
        if hasattr(self, "_prob"):
            del self._prob
        Nag = self._Pcore.shape[0]
        I = sps.eye(Nag, format='csc')
        lb, ub = (np.zeros(2*Nag), None)
        I = sps.bmat([[I,I],[-I,I]],'csc') # This captures the absolute value function
        q = np.concatenate((self.qhalf(col), 0.5*self.lsparse * np.ones(Nag)))
        prob = osqp.OSQP()
        prob.setup(self._P, q, I, lb, ub, verbose=False)
        return prob

    def solve(self, lsparse=None, lsmooth=None):
        if self.verbose: print("L1 sparsity reg. w/o nonnegative constraint")
        SpookQPBase.solve(self, lsparse, lsmooth)
        self.res = self.res[:self._Pcore.shape[0]] # the rest half are auxilary variables

    def update_lsparse(self, lsparse):
        # Updating lsparse just updates the q parameter in the problem
        self.lsparse = lsparse
        if not hasattr(self, "_probs"):
            qhalf = - self._Bcontracted.ravel()
            q = np.concatenate((qhalf, 0.5*self.lsparse * np.ones_like(qhalf)))
            self._prob.update(q = q)
        else:
            for col, prob in enumerate(self._probs):
                qhalf = - self._Bcontracted[...,col].ravel()
                q = np.concatenate((qhalf, 0.5*self.lsparse * np.ones_like(qhalf)))
                prob.update(q = q)
        if self.verbose: print("Sparsity hyperparam updated.")

    # def update_lsmooth(self, lsmooth):
    #     self.lsmooth = lsmooth
    #     Pnew = self.calcPtriu()
    #     self._update_Pmat(Pnew) 

