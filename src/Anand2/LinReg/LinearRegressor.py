# Please change the root variable as required.
# train and test directories of the dataset need to be present in the root directory
# weights need to be present in a weights folder in the root directory

from sklearn.utils import class_weight
import cv2, numpy as np
from sklearn.model_selection import StratifiedKFold,KFold,train_test_split

import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import cv2, pandas as pd
import numpy as np
import h5py
import pickle, math

from pandas.tools.plotting import autocorrelation_plot
#from statsmodels.graphics.gofplots import qqplot

from scipy.stats import probplot as qqplot

from sklearn.metrics import roc_auc_score, accuracy_score
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_squared_error as mse

np.random.seed(1)

from sklearn.linear_model import LinearRegression as LR


class Motor:
    def __init__(self):
        self.L = .5
        self.R = 1
        self.J = 0.01
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
	noisedot= np.zeros([1,2],dtype=float)
	noisedot[:,:]=np.random.normal(0.0, 1, 2)
	noise= np.zeros([1,2],dtype=float)
	noise[:,:]=np.random.normal(0.0, 1, 2)
	#print noisedot,noisedot.shape,self.stateDot.shape
	self.stateDot+=noisedot.transpose()
        self.state += self.stateDot.transpose() * self.dT
	self.state+=noise
        return self.state
    def update(self):
        self.C = np.array([1.0, 0])
        self.A = np.array([[0, 1.0], [-(self.R * self.b - self.Km * self.Kb) / (self.L * self.J),
                                      -(self.L * self.b + self.R * self.J) / (self.L * self.J)]])
        self.B = np.array([[0], [self.Km / (self.L * self.J)]])
        self.D = np.array([0.0])


def getControlInput():
    out = np.random.rand() * 20 - 10
    #out = np.random.rand() * 20
    return out

m = Motor()
dT = .001
m.setTimeStep(dT)
model = LR()
result = np.zeros([6,1])
printDuring=False

X=[]
y=[]

for i in np.arange(0, 50000):
    # StepWise control input function
    if ( np.mod(i,50) == 0 ):
        print "Loop-",i
	controlInput=0
	if i%1000==0 and i!=0:
        	controlInput = 5

    stateTensor =(m.state)
    stateTensor = np.concatenate((stateTensor,(np.ones([1,1], dtype=float) * controlInput)), 1)
    outBar=m.step(controlInput)
    if i<10000:
        out=np.zeros((1,2))
        X.append(stateTensor)
        y.append(outBar)
    elif i==10000:
        out=np.zeros((1,2))
        model.fit((np.asarray(X)).reshape(10000,3),(np.asarray(y)).reshape(10000,2))
    elif i>10000:
    	out=model.predict(stateTensor.reshape(1,3))
        model.fit(stateTensor.reshape(1,3),outBar)
    else:
	continue
    tmpResult = np.empty([6,1])
    tmpResult[0] = dT*(i+1)
    tmpResult[1] = out[0][0]
    tmpResult[2] = outBar[0][0]
    tmpResult[3] = out[0][1]
    tmpResult[4] = outBar[0][1]
    tmpResult[5] = controlInput
    result = np.concatenate((result,tmpResult),1)
    if np.mod(i,50) == 0 and printDuring:
        plt.figure(1)
        plt.clf()
        plt.subplot(2,1,1)
        plt.plot(result[0], result[1], result[0], result[2])#, result[0], result[5])
        plt.subplot(2,1,2)
        plt.plot(result[0], result[3], result[0], result[4])#, result[0], result[5])
        plt.draw()
        plt.pause(.001)


plt.figure(3)
plt.subplot(3, 1, 1)
plt.plot(result[0], result[1],c="k",linewidth="4",label="Pred")
plt.plot( result[0], result[2],c="r",label="True")
plt.ylabel("State 0")
plt.subplot(3, 1, 2)
plt.plot(result[0], result[3],c="k",linewidth="4",label="Pred")
plt.plot(result[0], result[4],c="r",label="True")
plt.ylabel("State 1")
plt.legend(loc="upper right")
plt.subplot(3, 1, 3)
plt.plot(result[0], result[5],c="k",label="Input")
plt.ylabel("Control Input")
plt.xlabel("Time")
plt.legend(loc="upper right")
plt.show()
plt.savefig("LinRegSystemidtest.svg")


#AutoCorrelation Residual Plot State-0

plt.clf()
residuals = pd.DataFrame((result[2]-result[1])[10001:])
autocorrelation_plot(residuals)
plt.show()
plt.savefig("LinRegAutoCorrelationState0.svg")
plt.clf()

#QQPlot Residuals State-0
qqplot((np.array(residuals)).flatten(),plot=plt)
plt.show()
plt.savefig("LinRegQQState0.svg")
plt.clf()

#AutoCorrelation Residual Plot State-1

residuals = pd.DataFrame((result[4]-result[3])[10001:])
autocorrelation_plot(residuals)
plt.show()
plt.savefig("LinRegAutoCorrelationState1.svg")
plt.clf()

#QQPlot Residuals State-1

qqplot((np.array(residuals)).flatten(),plot=plt)
plt.show()
plt.savefig("LinRegQQState1.svg")
plt.clf()

def NormCrossCorr(a,b,mode='same'):
        a = (a - np.mean(a)) / (np.std(a) * len(a))
        b = (b - np.mean(b)) / (np.std(b))
	print a,b
        c = np.correlate(a, b, mode)
	return c
#CrossCorrelation Check
plt.plot(NormCrossCorr(result[2,10001:],result[1,10001:],mode="same"),np.arange(-20000,20000),c="k",linewidth="4",label="CrossCorrelation")
plt.show()
plt.savefig("CrossCorrelationState0.svg")
plt.clf()

plt.plot(NormCrossCorr(result[4,10001:],result[3,10001:],mode="same"),np.arange(-20000,20000),c="k",linewidth="4",label="CrossCorrelation")
plt.show()
plt.savefig("CrossCorrelationState1.svg")
plt.clf()


print "Velocity Error:",np.sqrt(np.mean((result[1,10001:]-result[2,10001:])**2))
print "Acceleration Error:",np.sqrt(np.mean((result[3,10001:]-result[4,10001:])**2))


print "Linear Model Intercept", model.intercept_
print "Linear Model Weights",model.coef_
