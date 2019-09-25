"""
Double Pendulum Model
Kent Evans
2019-09-20

This outputs simulation data of a double pendulum from a given initial condition, sim time, and delta t
"""
import numpy as np
import time
import matplotlib.pyplot as plt
import datetime as dat


class DPend():
    def __init__(self, m1, m2, l1, l2, th1, th2, th1d, th2d, dt, t,  name=dat.datetime.now().strftime("%Y%m%d-%H%M%S")):
        self.m1 = m1
        self.m2 = m2
        self.l1 = l1
        self.l2 = l2
        self.th1 = th1
        self.th2 = th2
        self.th1d = th1d
        self.th2d = th2d
        self.th1dd = 0
        self.th2dd = 0
        self.dt = dt
        self.t = t
        self.name = name
        self.results = np.zeros([6, int(t/dt)])
        self.g = 9.81

    def getTh1dd(self):
        #return ( 1 / ( (self.m2*self.l1*np.power(np.cos(self.th1-self.th2),2)) - ((self.m1+self.m2)*self.l1) ) ) * ( (self.m2*self.l2*self.th2d*np.sin(self.th1-self.th2)) + (self.g*(self.m1+self.m2)*np.sin(self.th1)) + (self.m2*self.l2*np.cos(self.th1-self.th2))*( ( (self.m2*self.l1*np.power(self.th1d,2)*np.sin(self.th1-self.th2)) - (self.m2*self.g*np.sin(self.th2)) ) / (self.m2*self.l2) ))
        return ( ((self.m1+self.m2)*self.g*np.sin(self.th1)) - (self.m2*self.l2*np.power(self.th2d,2)*np.sin(self.th2-self.th1)) - ((self.m2*np.cos(self.th2-self.th1))*( (self.l1*np.power(self.th1d,2))*np.sin(self.th2-self.th1) + self.g*np.sin(self.th2) ) ) ) / ( (self.m2*self.l1*np.power(np.cos(self.th2-self.th1),2) - (self.m1+self.m2)*self.l1) )


    def getTh2dd(self):
        #return - ( (self.m2*self.l1*self.th1dd*np.cos(self.th1-self.th2)) - (self.m2*self.l1*np.power(self.th1d,2)*np.sin(self.th1-self.th2)) + (self.m2*self.g*np.sin(self.th2)) ) / (self.m2*self.l2)
        return ( (-self.l1*self.th1dd*np.cos(self.th2-self.th1)) - (self.l1*np.power(self.th1d,2)*np.sin(self.th2-self.th1)) - (self.g*np.sin(self.th2)) ) / (self.l2)

    def getM1X(self,th1):
        return self.l1*np.sin(th1)

    def getM1Y(self,th1):
        return -self.l1*np.cos(th1)

    def getM2X(self,th1,th2):
        return self.l1*np.sin(th1) + self.l2*np.sin(th2)

    def getM2Y(self,th1,th2):
        return -self.l1*np.cos(th1) - self.l2*np.cos(th2)

    def simulate(self):
        for i in range(self.results.shape[1]):
            print(i)
            #Save Results
            self.results[0,i] = self.th1
            self.results[1,i] = self.th2
            self.results[2,i] = self.th1d
            self.results[3,i] = self.th2d

            self.results[4,i] = self.th1dd
            self.results[5,i] = self.th2dd

            #Calc Next State
#            self.th1 += self.th1d * self.dt
#            self.th2 += self.th2d * self.dt
#            self.th1d += self.th1dd*self.dt
#            self.th2d += self.th2dd*self.dt
            self.th1dd = self.getTh1dd()
            self.th2dd = self.getTh2dd()
            self.th1d += self.th1dd *self.dt
            self.th2d += self.th2dd *self.dt
            self.th1 += self.th1d * self.dt
            self.th2 += self.th2d * self.dt

            #print(self.results[:, i])
        return

        #print(np.cos(self.th1-self.th2))

    def animate(self, step):
        fig = plt.figure()
        ax = fig.add_subplot(111)
        ax.set(xlim=(-10,10),ylim=(-10,0))
        x1 = self.getM1X(self.results[0,:])
        y1 = self.getM1Y(self.results[0,:])
        x2 = self.getM2X(self.results[0,:],self.results[1,:])
        y2 = self.getM2Y(self.results[0,:],self.results[1,:])

        for i in range(int(self.results.shape[1]/step)):
            if (i*step <= 1500):
                lowLim = 0
            else:
                lowLim = i*step - 1500

            x = [0, x1[i*step], x2[i*step]]
            y = [0, y1[i*step], y2[i*step]]

            plt.plot(x, y, 'ro-')
            ax.scatter(x1[lowLim:i*step], y1[lowLim:i*step], s = 1)
            ax.scatter(x2[lowLim:i*step], y2[lowLim:i*step], s = 1)
            ax.set(xlim=(-15, 15), ylim=(-15, 15))
            plt.pause(.001)
            ax.clear()

    def plot(self):
        fig1 = plt.figure()
        m1x = fig1.add_subplot(321)
        m1y = fig1.add_subplot(322)
        m2x = fig1.add_subplot(323)
        m2y = fig1.add_subplot(324)
        diffx = fig1.add_subplot(325)
        diffy = fig1.add_subplot(326)

        x1 = self.getM1X(self.results[0, :])
        y1 = self.getM1Y(self.results[0, :])
        x2 = self.getM2X(self.results[0, :], self.results[1, :])
        y2 = self.getM2Y(self.results[0, :], self.results[1, :])

        T = np.linspace(0, self.t-self.dt, self.t/self.dt)
        m1x.plot(T, x1)
        m1y.plot(T, y1)
        m2x.plot(T, x2)
        m2y.plot(T, y2)
        diffx.plot(T, x1-x2)
        diffy.plot(T, y1-y2)
        plt.show()

m1 = 2
m2 = 3
l1 = 2
l2 = 3

tester = DPend(m1,m2,l1,l2,np.pi/3,np.pi/2,0,0,.001,10,name= str(m1) + "-" + str(m2) + "-" + str(l1) + "-" + str(l2) + "_" + dat.datetime.now().strftime("%Y%m%d-%H%M%S"))
tester.simulate()
tester.plot()
tester.animate(30)

np.savetxt(tester.name+".csv", tester.results, fmt='%.18e', delimiter=',', newline='\n', header='', footer='', comments='# ', encoding=None)
