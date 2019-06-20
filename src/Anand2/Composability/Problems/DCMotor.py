# Please change the root variable as required.
# train and test directories of the dataset need to be present in the root directory
# weights need to be present in a weights folder in the root directory


import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import cv2, pandas as pd
import numpy as np
import pickle, math

np.random.seed(1)


class Motor:
    def __init__(self):
        self.L = .5
        self.R = 1
        self.J = 0.01
        self.b = .1
        self.Kb = .01
        self.Km = .01
        self.C = np.array([1.0, 0])
        self.A = np.array([[0, 1.0], [-(self.R * self.b - self.Km * self.Kb) / (self.L * self.J),
                                      -(self.L * self.b + self.R * self.J) / (self.L * self.J)]])
        self.B = np.array([[0], [self.Km / (self.L * self.J)]])
        self.D = np.array([0.0])
        self.dT = 0.0

        self.state = np.zeros([1,2], dtype=float)
        self.stateDot = np.zeros([1,2], dtype=float)

    def setTimeStep(self, dT):
        self.dT = dT
        return

    def step(self, u):
        self.stateDot = np.matmul(self.A, self.state.transpose()) + self.B * u
        noisedot= np.zeros([1,2],dtype=float)
        noisedot[:,:]=np.random.normal(0.0, 1, 2)
        noise= np.zeros([1,2],dtype=float)
        noise[:,:]=np.random.normal(0.0, 1, 2)
        #print noisedot,noisedot.shape,self.stateDot.shape
        #self.stateDot+=noisedot.transpose()
        self.state += self.stateDot.transpose() * self.dT
        #self.state+=noise*self.dT
        return self.state
    def update(self):
        self.C = np.array([1.0, 0])
        self.A = np.array([[0, 1.0], [-(self.R * self.b - self.Km * self.Kb) / (self.L * self.J),
                                      -(self.L * self.b + self.R * self.J) / (self.L * self.J)]])
        self.B = np.array([[0], [self.Km / (self.L * self.J)]])
        self.D = np.array([0.0])


def getControlInput():
	out = np.random.rand() * 20 - 10
	return out


