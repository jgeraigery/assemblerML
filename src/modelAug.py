import torch
import torch.nn as nn
import torch.nn.functional as f
import numpy as np
import matplotlib.pyplot as plt


class Motor:
    def __init__(self):
        self.L = .5
        self.R = 1
        self.J = .01
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
        #print(self.stateDot)
        self.state += self.stateDot.transpose() * self.dT
        return self.state

    def setParams(self, params):
        self.L = params[0]
        self.R = params[1]
        self.J = params[2]
        self.b = params[3]
        self.Kb = params[4]
        self.Km = params[5]
        return

class CorrNet(nn.Module):
    def __init__(self, num_inputs):
        super(CorrNet,self).__init__()
        self.num_inputs = num_inputs

        self.fc1 = nn.Linear(num_inputs,128)
        self.fc2 = nn.Linear(128,64)
        self.fc3 = nn.Linear(64, 3)

    def forward(self, inp):
        tmp = f.leaky_relu(self.fc1(inp))
        tmp = f.leaky_relu(self.fc2(tmp))
        return self.fc3(tmp)



motor = Motor()
model = Motor()
#set incorrect parameters and see if the neural network can correct for this
model.setParams([.1, 3, .1, .15, .005, .1])

#Neural Network Stuff
cn = CorrNet(3)
crit = nn.MSELoss()
optimizer = torch.optim.adam(cn.parameters(), lr=.0001)
