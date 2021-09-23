# -*- coding: utf-8 -*-
"""
Created on Tue Sep 19 19:42:45 2017

@author: mahbo
"""

import csv
import math
import os.path
import time
from collections import namedtuple

import numpy as np
import pandas as pd
import scipy.linalg as la

from model import model

kern = namedtuple('kern', 'name xbound ybound main side xthresh ythresh')


class problem:
    # A problem object is the main unit of action in this package. It encapsulates
    # the data for any problem, and is the conduit through which various fair and unfair
    # SVM and PCA algorithms can be run on the data, as well as the conduit through
    # which the results of these algorithms may be plotted and analyzed
    def __init__(self, filename=None, numFields=40, numPoints=500, corr=0.8, M=1000, perc=0, isGur=0) -> None:
        self.filename = filename
        self.M = 1000
        if self.filename is not None:
            if os.path.isfile(filename) and '.pickle' in filename:
                data = pd.read_pickle(filename)
            else:
                data = np.array([dPoint for dPoint in csv.reader(open(self.filename, 'r'))])
            self.headers = data[0].copy() if '.pickle' not in filename else data.columns.to_list()
            data = data[1:] if '.pickle' not in filename else np.array(data)
            self.data = data[:, :-2].copy().astype(float)
            self.mainResp = data[:, -2].copy().astype(float)
            # Demographic parity
            self.sideResp = data[:, -1].copy().astype(float)
            # Equalized opportunity
            # self.sideResp = (data[:, -1].copy().astype(bool) & data[:, -2].copy().astype(bool)).astype(float)
            self.numPoints, self.numFields = data.shape
            # self.data = normalize(self.data[1:,:-2].copy().astype(float))[0]
        else:
            self.numFields = numFields
            self.numPoints = numPoints
            B1 = np.random.random(numFields)
            B1 = B1 / la.norm(B1)
            B2 = np.random.random(numFields - 1)
            B2 = np.hstack((B2, -B1[:numFields - 1].dot(B2) / B1[numFields - 1]))
            B2 = B2 / la.norm(B2)
            B2 = corr * B1 + math.sqrt(1 - corr ** 2) * B2
            self.headers = ['X' + str(i) for i in range(1, numFields + 1)] + ['Y1', 'Y2']
            self.data = 10 * (2 * np.random.random((numPoints, numFields)) - 1)
            self.mainResp = np.array(
                [1 if np.random.random() <= math.exp(B1.dot(self.data[i,])) / (1 + math.exp(B1.dot(self.data[i,]))) \
                     else 0 for i in range(numPoints)]).astype(int)
            self.sideResp = np.array(
                [1 if np.random.random() <= math.exp(B2.dot(self.data[i,])) / (1 + math.exp(B2.dot(self.data[i,]))) \
                     else 0 for i in range(numPoints)]).astype(int)
        self.isSplit = False
        self.isGur = isGur
        if perc > 0:
            self.splitData(perc)

    def splitData(self, perc) -> None:
        # Splits data and main and side responses into training and testing sets
        if perc < 0 or perc > 1:
            print('Please enter a new percentage to use for training')
            return
        idx = np.arange(self.data.shape[0])
        np.random.shuffle(idx)
        trainidx = idx[:int(perc * self.data.shape[0])]
        testidx = idx[int(perc * self.data.shape[0]):]
        self.train = self.data[trainidx]
        self.test = self.data[testidx]
        self.trainMain = self.mainResp[trainidx]
        self.trainSide = self.sideResp[trainidx]
        self.testMain = self.mainResp[testidx]
        self.testSide = self.sideResp[testidx]
        self.numTrain = trainidx.size
        self.numTest = testidx.size
        self.isSplit = True

    def checkIfSplit(self, split, test=False, outputFlag=False):
        # Checks if the data is already split, and if not, splits data into 70% training
        # and 30% testing, as well as associated main and side responses
        if split:
            if not self.isSplit:
                if outputFlag:
                    print('Data not split, using standard 70-30 split')
                self.splitData(0.7)
            if test:
                return self.test, self.testMain, self.testSide, self.numTest
            else:
                return self.train, self.trainMain, self.trainSide, self.numTrain
        else:
            return self.data, self.mainResp, self.sideResp, self.numPoints

    def spca(self, dimPCA=2, d=0, mu=1, m=None, dualize=True, split=False, outputFlag=False, outputFull=False):
        # Runs the fair PCA with both constraints and returns both the optimal basis
        # vectors as well as the model object
        dat, mrsp, srsp, numPnt = self.checkIfSplit(split)

        totTime = time.time()
        if m is None:
            m = model(dat, dimPCA=dimPCA, outputFlag=outputFlag)
        m.addConstr(m.getZCon(srsp), rhs=d, record=False)
        m.addQuadConstr(srsp > 0, mu=mu, dualize=dualize)
        m.optimize(outputFlag=outputFlag)
        runtime = time.time() - totTime
        if outputFlag:
            print("Total running time: %s:%s" % (
                int((time.time() - totTime) / 60), round((time.time() - totTime) % 60, 2)))
        if outputFull:
            return m.B_full
        else:
            return m.B, m, runtime


def runSynthetic1() -> None:
    prob = problem('../synthetic1/synthetic1.csv')
    B, m, runtime = prob.spca(dimPCA=2, d=0, mu=0.01)

    np.savetxt('../synthetic1/FPCA_V_synthetic1.csv', B)

    return None


def runSynthetic2() -> None:
    num_repeats = 10

    runtimes = np.zeros((num_repeats, 9))

    tmp = 0
    for p in range(20, 110, 10):
        for repeat in range(num_repeats):
            print("p=%d, repeat=%d" % (p, repeat))
            prob = problem('../synthetic2/{}/train_{}.csv'.format(p, repeat + 1))
            B, m, runtime = prob.spca(dimPCA=5, d=0, mu=0.01)

            # Save runtime
            runtimes[repeat, tmp] = runtime

            # Save loading matrix
            np.savetxt('../synthetic2/{}/FPCA_V_{}.csv'.format(p, repeat + 1), B)
        tmp += 1
        np.savetxt('../synthetic2/FPCA_runtimes.csv', runtimes, delimiter=',')

    return None


def runUCI() -> None:
    dataset_names = ['COMPAS', 'German', 'Adult']
    # dataset_names = ['Adult']

    runtimes = np.zeros((10, 3))

    name_num = 0
    for name in dataset_names:
        for split in range(10):
            print("%s, repeat=%d" % (name, split))
            prob = problem('../datasets/{}/train_{}.csv'.format(name, split))
            # B, m, runtime = prob.spca(dimPCA=10, d=0, mu=0.01)
            B, m, runtime = prob.spca(dimPCA=2, d=0.1, mu=0.01)

            # Save runtime
            runtimes[split, name_num] = runtime

            # Save loading matrix
            np.savetxt('../uci/{}/FPCA_V_{}.csv'.format(name, split), B, delimiter=',')

        name_num += 1
        np.savetxt('../uci/fpca/runtimes.csv'.format(name), runtimes, delimiter=',')

    return None


def runCompare() -> None:
    # dataset_names = ['COMPAS', 'German', 'Adult']
    name = 'German'

    for split in range(10):
        print("%s, repeat=%d" % (name, split))
        prob = problem('../datasets/{}/train_{}.csv'.format(name, split))
        P = prob.spca(dimPCA=10, d=0, mu=0.01, outputFull=True)

        # Save loading matrix
        np.savetxt('../uci/supplementary-explanation/FPCA_P_{}.csv'.format(split), P, delimiter=',')

    return None


if __name__ == '__main__':
    # runSynthetic1()
    # runSynthetic2()
    # runUCI()
    runCompare()
