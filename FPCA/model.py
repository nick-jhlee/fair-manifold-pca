# -*- coding: utf-8 -*-
"""
Created on Thu Oct 26 18:25:32 2017

@author: mahbo

"""

from mosPCA import *
from mosPCAMult import *


class model:
    # A model object is the conduit through which all optimization is done. It stand to
    # handle the different types of optimization objects from a unified framework from
    # the perspective of the problem object. It has the added benefit of being a self
    # contained optimization, allowing for cross-validation or comparisons at the level
    # of the problem object without having to define all of the extra variables and
    # constraint coefficients associated with each individual optimization
    def __init__(self, dat, rsp=None, lam=None, conic=True, dual=False, kernel=None, dimPCA=None,
                 lossneg=None, losspos=None, lasso=False, EDE=False, outputFlag=False, manifold=False):
        self.numPoints = dat.shape[0]
        self.numFields = dat.shape[1]
        if kernel is None:
            kernel = lambda x, y: x.T.dot(y)
        self.kernel = kernel
        self.Status = "unsolved"
        self.RunTime = 0
        self.rsp = rsp
        self.conic = conic
        self.dual = dual
        self.B = np.zeros((self.numFields, 1))
        self.B_full = np.zeros((self.numFields, self.numFields))
        self.B_eig = 0
        self.b = 0
        self.ObjVal = 0
        self.numProjCons = 0
        self.K = None
        self.Y = None
        self.lossneg = lossneg
        self.losspos = losspos
        self.lasso = lasso
        self.EDE = EDE
        self.dimPCA = dimPCA
        self.lam = lam
        if self.dual:
            self.K = np.empty((self.numPoints, self.numPoints))
            self.K[np.tril_indices(self.numPoints)] = [kernel(dat[i], dat[j]) for i in range(self.numPoints) for j in
                                                       range(i + 1)]
            self.K += np.tril(self.K, -1).T
            self.Y = rsp[:, None].dot(rsp[None, :])
            if np.sum(np.isnan(self.K)) > 0:
                self.K[np.where(np.isnan(self.K))] = 0.9995
            if np.sum(np.isinf(self.K)) > 0:
                print(np.where(np.isinf(self.K)))
        if dimPCA is None or dimPCA == 1:
            self.m = mosPCA(dat, outputFlag=outputFlag)
        else:
            self.m = mosPCAMult(dat, dimPCA, outputFlag=outputFlag, manifold=manifold)

    def kFold(self, k=5):
        # Splits data into k folds
        idx = np.arange(self.numPoints)
        np.random.shuffle(idx)
        folds = [idx[int(i * self.numPoints / k):int((i + 1) * self.numPoints / k)] for i in range(k)]
        return folds

    def optimize(self, outputFlag=False) -> None:
        # Runs the optimization procedure
        self.m.optimize()
        self.RunTime = self.m.RunTime
        self.B = np.array(self.m.getB())
        self.B_full = np.array(self.m.getB_full())
        self.B_eig = self.m.getB_eig()
        if len(self.B.shape) == 1:
            self.B = self.B.reshape((len(self.m.getB()), 1))
        self.ObjVal = self.m.getObj()
        if outputFlag:
            print("Optimization time: %s" % (round(self.RunTime, 2)))

    def getStatus(self):
        # Returns the status of the optimizer
        stat = self.m.task.getsolsta(mosek.soltype.itr)
        # if stat in [mosek.solsta.optimal, mosek.solsta.near_optimal]:
        if stat in [mosek.solsta.optimal]:
            return 'optimal'
        # elif stat in [mosek.solsta.prim_infeas_cer, mosek.solsta.near_prim_infeas_cer]:
        elif stat in [mosek.solsta.prim_infeas_cer]:
            return 'infeasible'
        # elif stat in [mosek.solsta.dual_infeas_cer, mosek.solsta.near_dual_infeas_cer]:
        elif stat in [mosek.solsta.dual_infeas_cer]:
            return 'unbounded'
        else:
            return 'other'

    def getRHS(self):
        # Returns vector of right-hand-sides (ONLY WORKS FOR MOSEK)
        return self.m.getRHS()

    def getZCon(self, rsp):
        # Returns the coefficients of the mean constraint
        rsp = rsp > 0
        f = self.rsp * (np.mean(self.K[rsp], axis=0) - np.mean(self.K[~rsp], axis=0)) if self.dual \
            else np.mean(self.m.dat[rsp], axis=0) - np.mean(self.m.dat[~rsp], axis=0)
        return f

    def getSig(self, rsp):
        # Return the mean-normalized covariance matrix
        mat = self.K if self.dual else self.m.dat
        rsp = rsp > 0
        return mat[rsp].T.dot(np.eye(sum(rsp)) - np.ones((sum(rsp), sum(rsp))) / sum(rsp)).dot(mat[rsp]) / sum(rsp)

    def addConstr(self, coeff, rhs=0, record=True) -> None:
        # Handles the addition of a single linear constraint and records it
        self.m.addConstr(np.tensordot(coeff, coeff, axes=0), rhs ** 2)
        if record:
            self.numProjCons += 1

    def addQuadConstr(self, rsp, mu=1, dualize=False):
        # Handles the addition of a single covariance constraint (ONLY FOR MOSEK)
        self.m.addQuadConstr(self.getSig(rsp > 0) - self.getSig(rsp <= 0), mu=mu, dualize=dualize)

    def projCons(self, projMat=None) -> None:
        # Projects all data according to given matrix and updates optimization model
        # (ONLY FOR MOSEK)
        self.m.projCons(projMat)
