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


class Two_Tank:
    def __init__(self):
        self.input_size = 3
        self.output_size = 2
        self.A1 = 15.38				#Tank Cross Sectional Area
        self.A2 = 15.38

        self.Ao1 = 0.1781			#Tank Orifice Cross Sectional Area
        self.Ao2 = 0.1781
        self.Kp = 4.6				#Pump constant
	self.g=g*100				#Gravitational constant

        self.C = np.array([1.0, 0])
        self.A = np.array([[-self.Ao1/self.A1, 0], [self.Ao1/self.A2,-self.Ao2/self.A2]])
        self.B = np.array([[self.Kp / (self.A1)],[0]])
        self.D = np.array([0.0])

        self.dT = 0.0

        self.state = np.zeros([1,2], dtype=float)
        self.stateDot = np.zeros([1,2], dtype=float)
	self.state[:,:]=0.00001
	self.stateDot[:,:]=0.00001
    def setTimeStep(self, dT):
        self.dT = dT
        return

    def step(self, u):
        self.stateDot = np.matmul(self.A, np.sqrt(2*self.g*self.state.transpose())) + self.B * u
        self.state += self.stateDot.transpose() * self.dT
        return self.state
    def update(self):
        self.C = np.array([1.0, 0])
        self.A = np.array([[-self.Ao1/self.A1, 0], [self.Ao1/self.A2,-self.Ao2/self.A2]])
        self.B = np.array([[self.Kp / (self.A1)],[0]])
        self.D = np.array([0.0])


    def getControlInput(self):
	out = np.random.rand() * 10
	return out


    def reset(self):
        self.C = np.array([1.0, 0])
        self.A = np.array([[-self.Ao1/self.A1, 0], [self.Ao1/self.A2,-self.Ao2/self.A2]])
        self.B = np.array([[self.Kp / (self.A1)],[0]])
        self.D = np.array([0.0])

        self.dT = 0.0

        self.state = np.zeros([1,2], dtype=float)
        self.stateDot = np.zeros([1,2], dtype=float)
	self.state[:,:]=0.00001
	self.stateDot[:,:]=0.00001
