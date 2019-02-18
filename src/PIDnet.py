#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Feb 16 19:25:54 2019

@author: Kent Evans

The idea here is to have a neural net control the gains of a PID controller.
Essentially, the idea here is to introduce some feed forward control by 
having the net learn on current state, desired state for the next N time steps
and various gains.

The loss function will, at least initially, be defined as the sum of the MSE between the
setpoint and trajectory at every point in time.

I don't think this will work for any arbitrary path.  We will have to train for 
every unique path?  Don't see a clear way forward for that.

Network architecture:
   
INPUT                   HIDDEN              OUTPUT
   x|t                                      Kp|t+1
   xdot|tC                                  Ki|T+1
   r|t+1                                    Kd|t+1
   rdot|t+1          FC 64 - FC 3
   ...
   r|t+L+1
   rdot|t+L+1  (r/rdot)
   Kp|t
   Ki|t   
   Kd|t 
    

"""

import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import numpy as np

class motor:
    def __init__(self , tC ):
        self.L = .5
        self.R = 1
        self.J = .01
        self.b = .1
        self.Kb = .01
        self.Km = .01
        self.C = np.array([1.0,0])
        self.A = np.array([[0,1.0],[-(self.R*self.b-self.Km*self.Kb)/(self.L*self.J),-(self.L*self.b+self.R*self.J)/(self.L*self.J)]])
        self.B = np.array([[0],[self.Km/(self.L*self.J)]])
        self.D = np.array([0.0])
        self.tC = tC
        self.dTc = tC[1]-tC[0]
        
        """
        self.tL = tC[0::10] #Neural net rate is 1/10 that of the control loop
        self.dTl = self.tL[1]-self.tL[0]
        """
        
        self.state = np.zeros((2,1))
        self.gains = np.ones((3,len(self.tC)))
        
class pidRNN(nn.Module):
    def __init__(self):
        super(pidRNN,self).__init__(mot)
        self.hiddenSize = 64
        
    def forward(self):
        project = True
        
t = np.arange(0,60,.01)
mot = motor(t)
r = np.sin(t)

net = pidRNN(mot)