#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan 30 18:36:59 2019

@author: Kent Evans

    Moderately comfortable with pytorch.  Starting to put together our learning
    algorithm.  Attempting to learn a PID control system by taking the inputs 
    to the outputs and jacobians, then using a neural net to go backwards to
    the input.  We can then use this learned neural net to do backprop in a 
    larger net.
    
    To start, will just model a dc motor speed controller.  Mass attached?
    
"""

import torch
import torch.nn as nn
import torchvision as tv
import numpy as np
import scipy.signal as sci
import matplotlib.pyplot as plt

class system:
    def __init__ ( self ):
        self._j = .01
        self._k = .01
        self._kb = .01
        self._b = .1
        self._l = .5
        self._r = 1
        self.delT = .01
        self._t = (0,10,self.delT)
        #self._y = .015891 - (.01804*np.exp(-2.0026*t)) + (2.39e5)*np.sin(6.283*t) + .001127*np.exp(-9.9975*t)

        
    def plot ( self ):
        
    
    def transfer ( self , input ):
        
    
class controller:
    def __init__(self):
        self._kp = .2
        self._kd = .05
        self._ki = .02
        self.intErr = 0
        self.lastE = 0
        self.delT = .01
        
    def setGains ( self , gains ):
        self._kp = gains[0]
        self._kd = gains[1]
        self._ki = gains[2]
    
    def mapping ( self , error ):
        self.intErr += error*self.delT
        out = self._kp*error + self._kd*((error - self.lastE)/self.delT) + self._ki*(self.intErr)
        self.lastE = error
        return out
        
#Main Code starts here
        
