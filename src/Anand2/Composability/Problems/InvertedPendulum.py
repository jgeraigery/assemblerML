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


class Inverted_Pendulum:
    def __init__(self):
	self.input_size=5
	self.output_size=4
        self.M = .5			#Mass of Cart (kg)
        self.m = 0.1			#Mass of Pendulum (kg)
        self.b = .1			#Coefficient of Friction N*s/m
        self.L = .3			#Length of Pendulum (m)
        self.I = 0.005			#Mass Moment of Inertia (kg*m**2)

	self.p=self.I*(self.M+self.m)+self.M*self.m*(self.L)**2 #Denominator Terms in A and B Matrices

        self.C = np.array([[1.0, 0, 0, 0],[0, 0, 1, 0]])
	self.A = np.array([[0, 1, 0, 0],[0, -(self.I+self.m*self.L**2)*self.b/self.p, (self.m**2*g*self.L**2)/self.p, 0],
     			[0, 0, 0, 1],[0, -(self.m*self.L*self.b)/self.p, self.m*g*self.L*(self.M+self.m)/self.p, 0]])

        self.B = np.array([[0], [(self.I+self.m*self.L**2)/self.p], [0], [self.m*self.L/self.p]])
        self.D = np.array([[0.0],[0.0]])

        self.dT = 0.0

        self.state = np.zeros([1,4], dtype=float)
        self.stateDot = np.zeros([1,4], dtype=float)

    def setTimeStep(self, dT):
        self.dT = dT
        return

    def step(self, u):
	self.stateDot = np.matmul(self.A, self.state.T) + self.B * u
        noisedot= np.zeros([1,4],dtype=float)
        noisedot[:,:]=np.random.normal(0.0, 1, 4)
        noise= np.zeros([1,4],dtype=float)
        noise[:,:]=np.random.normal(0.0, 1, 4)
	self.state += self.stateDot.transpose() * self.dT

        return self.state
    def update(self):

	self.p=self.I*(self.M+self.m)+self.M*self.m*(self.L)**2 #Denominator Terms in A and B Matrices

	self.A = np.array([[0, 1, 0, 0],[0, -(self.I+self.m*self.L**2)*self.b/self.p, (self.m**2*g*self.L**2)/self.p, 0],
     			[0, 0, 0, 1],[0, -(self.m*self.L*self.b)/self.p, self.m*g*self.L*(self.M+self.m)/self.p, 0]])

        self.B = np.array([0, (self.I+self.m*self.L**2)/self.p, 0, self.m*self.L/self.p]).T


    def getControlInput(self):
	out = np.random.rand() *0.002  - 0.001
	return out


