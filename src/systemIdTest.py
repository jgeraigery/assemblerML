"""
Created: 2019-04-02
Author: Kent Evans



"""

import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt

plt.ion()

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


class IdNet(nn.Module):
    def __init__(self, num_inputs):
        super(IdNet, self).__init__()
        self.num_inputs = num_inputs
        self.fc1 = nn.Linear(self.num_inputs, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 2)

    def forward(self, inp):
        tmp = F.leaky_relu(self.fc1(inp))

        tmp = F.leaky_relu(self.fc2(tmp))

        return self.fc3(tmp)


def getControlInput():
    out = np.random.rand() * 20 - 10
    return out





m = Motor()
dT = .001
m.setTimeStep(dT)
sn = IdNet(3)
criterion = nn.MSELoss()
optimizer = torch.optim.SGD(sn.parameters(), lr=.00001, momentum=0.5)
result = np.zeros([6,1])
sn.zero_grad()
printDuring = False

for i in np.arange(0, 1000000):
    # control input function
    if ( np.mod(i,200) == 0 ):
        controlInput = getControlInput()

    stateTensor = torch.from_numpy(m.state)
    # print(stateTensor)
    stateTensor = torch.cat((stateTensor, torch.from_numpy(np.ones([1,1], dtype=float) * controlInput)), 1).float()
    #print(stateTensor)
    # print(type(stateTensor))
    out = sn.forward(stateTensor)
    # print(out)
    outBar = m.step(controlInput)
    outBar = torch.from_numpy(outBar).float()
    #Store result
    tmpResult = np.empty([6,1])
    tmpResult[0] = dT*(i+1)
    #print(out[0][0].item())
    tmpResult[1] = out[0][0].item()
    tmpResult[2] = outBar[0][0].item()
    tmpResult[3] = out[0][1].item()
    tmpResult[4] = outBar[0][1].item()
    tmpResult[5] = controlInput
    result = np.concatenate((result,tmpResult),1)
    loss = criterion(outBar, out)
    #print(loss)
    if ( np.mod(i,5)) == 0:
        #sn.zero_grad()
        optimizer.zero_grad()
        pass

    # backward passes accumulate gradients, need to zero them each time (unless it's an RNN)
    loss.backward()
    nn.utils.clip_grad_norm_(sn.parameters(),10)
    #Clip gradient here?
    optimizer.step()
    if np.mod(i,100) == 0 and printDuring:
        plt.figure(1)
        plt.clf()
        plt.subplot(2,1,1)
        plt.plot(result[0], result[1], result[0], result[2])
        plt.subplot(2,1,2)
        plt.plot(result[0], result[3], result[0], result[4])
        plt.draw()
        plt.pause(.001)
plt.ioff()
plt.figure(2)
plt.subplot(2, 1, 1)
plt.plot(result[0], result[1], result[0], result[2])
plt.subplot(2, 1, 2)
plt.plot(result[0], result[3], result[0], result[4])
plt.show()