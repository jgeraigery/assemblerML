# Please change the root variable as required.
# train and test directories of the dataset need to be present in the root directory
# weights need to be present in a weights folder in the root directory

from sklearn.utils import class_weight
<<<<<<< HEAD
=======
from keras import applications
from keras.models import Sequential
from keras.layers.core import Flatten, Dense, Dropout
from keras.layers.convolutional import Convolution2D, MaxPooling2D, ZeroPadding2D
from keras.optimizers import SGD
>>>>>>> cf7c18ea2072e20a32a5f15aecd1da54d5f6ea18
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

<<<<<<< HEAD
from pandas.tools.plotting import autocorrelation_plot
#from statsmodels.graphics.gofplots import qqplot

from scipy.stats import probplot as qqplot

from sklearn.metrics import roc_auc_score, accuracy_score
from sklearn.preprocessing import LabelEncoder
=======
import keras
from keras.layers.advanced_activations import LeakyReLU, PReLU,ELU
from keras.preprocessing.image import ImageDataGenerator
from keras.layers import Convolution2D, Dense, Input, MaxPooling2D, Dropout, Flatten, ZeroPadding2D, Activation,LSTM,Bidirectional,Convolution1D,MaxPooling1D,Conv1D,SimpleRNN,Lambda
from keras.layers.pooling import AveragePooling2D,GlobalAveragePooling1D
from keras.models import Model, Sequential
from keras.utils.vis_utils import plot_model
from keras.optimizers import SGD, Adam
from keras.regularizers import l2
from keras.callbacks import ModelCheckpoint, LearningRateScheduler, ReduceLROnPlateau,EarlyStopping
from sklearn.metrics import roc_auc_score, accuracy_score
import keras.backend as K
from keras.preprocessing.image import ImageDataGenerator
from keras import regularizers
from keras.metrics import categorical_accuracy
from sklearn.preprocessing import LabelEncoder
from keras.utils import np_utils
>>>>>>> cf7c18ea2072e20a32a5f15aecd1da54d5f6ea18
from sklearn.metrics import mean_squared_error as mse

np.random.seed(1)

from sklearn.linear_model import LinearRegression as LR


class Motor:
    def __init__(self):
        self.L = .5
        self.R = 1
<<<<<<< HEAD
        self.J = 0.01
=======
        self.J = 1.0
>>>>>>> cf7c18ea2072e20a32a5f15aecd1da54d5f6ea18
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
<<<<<<< HEAD
	noisedot= np.zeros([1,2],dtype=float)
	noisedot[:,:]=np.random.normal(0.0, 1, 2)
	noise= np.zeros([1,2],dtype=float)
	noise[:,:]=np.random.normal(0.0, 1, 2)
	#print noisedot,noisedot.shape,self.stateDot.shape
        #print(self.stateDot)
	self.stateDot+=noisedot.transpose()
        self.state += self.stateDot.transpose() * self.dT
	self.state+=noise
=======
        #print(self.stateDot)
        self.state += self.stateDot.transpose() * self.dT
>>>>>>> cf7c18ea2072e20a32a5f15aecd1da54d5f6ea18
	#print self.state
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
<<<<<<< HEAD
    # StepWise control input function
=======
    # control input function
>>>>>>> cf7c18ea2072e20a32a5f15aecd1da54d5f6ea18
    if ( np.mod(i,50) == 0 ):
        print "Loop-",i
	controlInput=0
	if i%1000==0 and i!=0:
<<<<<<< HEAD
        	controlInput = 5

=======
        	#controlInput = getControlInput()
        	controlInput = 5
    #if (i%10000==0):
	m.J=m.J-0.1
	m.update()
>>>>>>> cf7c18ea2072e20a32a5f15aecd1da54d5f6ea18
    stateTensor =(m.state)
    stateTensor = np.concatenate((stateTensor,(np.ones([1,1], dtype=float) * controlInput)), 1)
    outBar=m.step(controlInput)
    if i<10000:
        out=np.zeros((1,2))
        X.append(stateTensor)
        y.append(outBar)
    elif i==10000:
        out=np.zeros((1,2))
<<<<<<< HEAD
        model.fit((np.asarray(X)).reshape(10000,3),(np.asarray(y)).reshape(10000,2))
    elif i>10000:
    	out=model.predict(stateTensor.reshape(1,3))
        model.fit(stateTensor.reshape(1,3),outBar.reshape(1,2))
=======
    	#model.fit(stateTensor.reshape(1,3),outBar)
        model.fit((np.asarray(X)).reshape(10000,3),(np.asarray(y)).reshape(10000,2))
    elif i>10000:
    	out=model.predict(stateTensor.reshape(1,3))
    	model.fit(stateTensor.reshape(1,3),outBar)
>>>>>>> cf7c18ea2072e20a32a5f15aecd1da54d5f6ea18
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

<<<<<<< HEAD

#AutoCorrelation Residual Plot State-0

plt.clf()
residuals = pd.DataFrame((result[2]-result[1])[10000:])
autocorrelation_plot(residuals)
plt.show()
plt.savefig("LinRegAutoCorrelationState0.svg")
plt.clf()

#QQPlot Residuals State-0
qqplot((np.array(residuals)).reshape(40001),plot=plt)
plt.show()
plt.savefig("LinRegQQState0.svg")
plt.clf()

#AutoCorrelation Residual Plot State-1

residuals = pd.DataFrame((result[4]-result[3])[10000:])
autocorrelation_plot(residuals)
plt.show()
plt.savefig("LinRegAutoCorrelationState1.svg")
plt.clf()

#QQPlot Residuals State-1

qqplot((np.array(residuals)).reshape(40001),plot=plt)
plt.show()
plt.savefig("LinRegQQState1.svg")
plt.clf()

#Sanity Check

qqplot((np.array(result[4])[10000:]).reshape(40001),plot=plt)
plt.show()
plt.savefig("LinRegQQCheck1.svg")
plt.clf()

#Sanity Check

qqplot(np.random.normal(0,1,1000),plot=plt)
#plt.show()
plt.savefig("LinRegQQCheck2.svg")
plt.clf()



=======
>>>>>>> cf7c18ea2072e20a32a5f15aecd1da54d5f6ea18
print "Velocity Error:",np.sqrt(np.mean((result[1,10000:]-result[2,10000:])**2))
print "Acceleration Error:",np.sqrt(np.mean((result[3,10000:]-result[4,10000:])**2))


<<<<<<< HEAD
print "Linear Model Intercept", model.intercept_
print "Linear Model Weights",model.coef_
=======
np.save("LinRegresult.npy",result)
print m.b

print model.intercept_
print model.coef_
>>>>>>> cf7c18ea2072e20a32a5f15aecd1da54d5f6ea18
