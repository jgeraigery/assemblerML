# -*- coding: utf-8 -*-
"""
Created on Thu Feb  7 09:28:19 2019

@author: Kent Evans



"""
import scipy.linalg as la
import numpy as np
import matplotlib.pyplot as plt

class motor:
    def __init__(self):
        self.L = .5
        self.R = 1
        self.J = .01
        self.b = .1
        self.Kb = .01
        self.Km = .01
        self.C = np.array([1.0,0,0])
        self.A = np.array([[0,1.0,0],[-(self.R*self.b-self.Km*self.Kb)/(self.L*self.J),-(self.L*self.b+self.R*self.J)/(self.L*self.J),0],self.C])
        self.B = np.array([[0],[self.Km/(self.L*self.J)],[0.0]])
        self.D = np.array([0.0])
        self.Q = np.diag([1000,0,100])
        self.Rm = np.array([.00001])
        self.H = np.concatenate((np.concatenate((self.A,-self.B*np.array(1/self.Rm[0])*np.transpose(self.B)),axis = 1),np.concatenate((self.Q,-self.A.T),axis = 1)),axis = 0)
        #[self.T,self.Z,numEigs] = la.schur(self.H, output = 'complex',sort = "lhp")
        #[self.W,self.Y,numEigs] = la.schur(self.H, output = 'real',sort = "lhp")
        #[self.T,self.Z,numEigs] = la.schur(self.H, sort = "rhp")
        #[self.T,self.Z] = la.schur(self.H, sort = None)
        self.X = la.solve_continuous_are(self.A,self.B,self.Q,self.Rm)
        self.K = np.matmul(np.array(1/self.Rm[0])*self.B.T,self.X)
        self.state = np.zeros([3,1])
        self.input = np.zeros([1,1])
        
    def genRefTraj(self,t,y):
        self.t = t
        self.dT = t[1]-t[0]
        self.xd = np.zeros_like(t)
        
        for i in range(len(t)):
            self.xd[i] = self.ud(t[i])
        
        #self.xd = y
        self.ud = np.array(self.xd*(self.R*self.b-self.Km*self.Kb)/self.Km)
        #self.ud = np.zeros_like(self.xd)
        self.xd = np.array([self.xd,np.zeros(self.xd.shape),np.zeros(self.xd.shape)])
       
    def simulate(self):
        self.stateRes = np.zeros([self.xd.shape[0],len(self.t)])
        for i in range(len(self.t)):
            statedot = np.matmul(self.A,self.state) + np.matmul(self.B,self.getInput(i)) - np.array([[0],[0],[1]])*self.xd[0,i]
            self.state += statedot*self.dT
            self.stateRes[:,i,None] = self.state
        plt.plot(self.t,self.stateRes[0,:],self.t,self.xd[0,:])
        #plt.axis([0,1,-1.2,1.2])
        #plt.show()
    
    def ud(self,t):
        if (t < 1):
            return 1
        elif (t<2):
            return 5
        elif (t<3):
            return 3
        elif (t<4):
            return 9
        elif (t<5):
            return 7
        elif (t<6):
            return 5
        elif (t<7):
            return 1
        else:
            return 0
        
    def getInput(self,ind):
        #print(np.matmul(-self.K,self.state-self.xd[:,1,None]) + self.ud[ind])
        return np.matmul(-self.K,self.state-self.xd[:,ind,None]) + self.ud[ind]
        
a = motor()
t = np.arange(0,10,.001)
y = np.sin(2*np.pi*4*t)
#y = np.ones_like(t)
a.genRefTraj(t,y)
a.simulate()
