# -*- coding: utf-8 -*-
"""
Created on Thu Feb  7 09:28:19 2019

@author: Kent Evans



"""
import scipy.linalg as la
import numpy as np

class motor:
    def __init__(self):
        self.L = .5
        self.R = 1
        self.J = .01
        self.b = .1
        self.Kb = .01
        self.Km = .01
        self.A = np.array([[0,1.0,0],[-(self.R*self.b-self.Km*self.Kb)/(self.L*self.J),-(self.L*self.b+self.R*self.J)/(self.L*self.J),0],[0,1,0]])
        self.B = np.array([[0],[self.Km/(self.L*self.J)],[0.0]])
        self.C = np.array([[0],[1.0],[0]])
        self.D = np.array([0.0])
        self.Q = np.diag([1,1])
        self.R = np.array([1])
        