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
from scipy.constants import g

np.random.seed(1)


class NNDelay:
    def __init__(self):
        self.input_size = 2
        self.output_size = 1
        self.K = 0.4				#Constant K
        self.T=0.9
        self.TD=5.0

	self.t=0.0
        self.dT = 0.0

	self.A=np.array([-1/self.T])
	self.B=np.array([self.K*(self.t-self.TD)])

        self.state = np.zeros((1,1))
        self.stateDot = np.zeros((1,1))

    def setTimeStep(self, dT):
        self.dT = dT
        return

    def step(self, u):
	self.t+=self.dT
	self.B=np.array([self.K*(self.t-self.TD)])
        self.stateDot = np.matmul(self.A,self.state)+self.B*u
        self.state += self.stateDot*self.dT
        return np.array(self.state)
    def update(self):
	self.A=np.array([-1/self.T])
	self.B=np.array([self.K*(self.t-self.TD)])
	return

    def getControlInput(self):
	out = np.random.rand() * 10
	return out


    def reset(self):
	self.t=0.0
        self.dT = 0.0

	self.A=np.array([-1/self.T])
	self.B=np.array([self.K*(self.t-self.TD)])

        self.state = np.zeros((1,1))
        self.stateDot = np.zeros((1,1))

	return

