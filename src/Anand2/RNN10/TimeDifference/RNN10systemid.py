# Please change the root variable as required.
# train and test directories of the dataset need to be present in the root directory
# weights need to be present in a weights folder in the root directory

from sklearn.utils import class_weight
from keras import applications
from keras.models import Sequential
from keras.layers.core import Flatten, Dense, Dropout
from keras.layers.convolutional import Convolution2D, MaxPooling2D, ZeroPadding2D
from keras.optimizers import SGD
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
from sklearn.metrics import mean_squared_error as mse, r2_score as r2

np.random.seed(1)

def NormCrossCorr(a,b,mode='same'):
        a = (a - np.mean(a)) / (np.std(a) * len(a))
        b = (b - np.mean(b)) / (np.std(b))
        c = np.correlate(a, b, mode)
        return c



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
        noisedot[:,:]=np.random.normal(0.0, 0.05, 2)
        noise= np.zeros([1,2],dtype=float)
        noise[:,:]=np.random.normal(0.0, 0.05, 2)
        #print noisedot,noisedot.shape,self.stateDot.shape
        #print(self.stateDot)
        #self.stateDot+=noisedot.transpose()
        self.state += self.stateDot.transpose() * self.dT
        #self.state+=noise
        #print (self.state)
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


def densemodel():
        input = Input(batch_shape=(None,3))
        x=Dense(100,activation="relu")(input)
        x=Dense(100,activation="relu")(x)
        output = Dense(2)(x)
        model = Model(inputs=input, outputs=output)
        model.compile(loss="mse", optimizer='adam', metrics=['accuracy'])
        return model


def rnnmodel():
        input = Input(batch_shape=(None,10,3))
        x=SimpleRNN(10,activation="relu",return_sequences=True)(input)
        x=SimpleRNN(10,activation="relu")(input)
        x = Dense(10,activation="relu")(x)
        output = Dense(2)(x)
        model = Model(inputs=input, outputs=output)
	adam=keras.optimizers.Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False)
        model.compile(loss="mse", optimizer=adam, metrics=['accuracy'])
        return model


m = Motor()
dT = .001
m.setTimeStep(dT)
model = rnnmodel()
result = np.zeros([8,1])
printDuring=False

X=[]
y=[]
movingvalue=np.zeros((10,3))

for i in np.arange(0, 50000):
    # control input function
    if ( np.mod(i,50) == 0 ):
        print ("Loop-",i)
	controlInput=0
	if i%1000==0 and i!=0:
        	controlInput = 15
        	#controlInput = getControlInput()
    if (i%10000==0):
	#m.J=m.J+0.1
	m.update()
    stateTensor =(m.state)
    stateTensor = np.concatenate((stateTensor,(np.ones([1,1], dtype=float) * controlInput)), 1)
    outBar=m.step(controlInput)
    movingvalue[0:9,:]=movingvalue[1:10,:]
    movingvalue[9,:]=stateTensor
    movingvalue2=np.asarray(list(movingvalue))

    print ("Current State, Next State, True Difference-",stateTensor[:,0:2],outBar,outBar-stateTensor[:,0:2])

    if i<10000:
        out=np.zeros((1,2))
        X.append(movingvalue2)
        y.append(outBar-stateTensor[:,0:2])
    elif i==10000:
        out=np.zeros((1,2))
        model.fit(np.asarray(X),(np.asarray(y)).reshape(10000,2),epochs=30)
	adam=keras.optimizers.Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False)
	model.compile(loss="mse", optimizer=adam, metrics=['accuracy'])
    elif i>10000:
    	out=model.predict(movingvalue2.reshape(1,10,3))
    	model.fit(movingvalue2.reshape(1,10,3),outBar-stateTensor[:,0:2],epochs=1)
    else:
	continue
    tmpResult = np.empty([8,1])
    tmpResult[0] = dT*(i+1)
    tmpResult[1] = out[0][0]
    tmpResult[2] = outBar[0][0]
    tmpResult[3] = out[0][1]
    tmpResult[4] = outBar[0][1]
    tmpResult[5] = controlInput
    tmpResult[6] = stateTensor[0][0]
    tmpResult[7] = stateTensor[0][1]
    result = np.concatenate((result,tmpResult),1)


plt.figure(3)
plt.subplot(3, 1, 1)
plt.plot(result[0,:10001], result[1,:10001],c="k",linewidth="4")
plt.plot(result[0,10001:], result[1,10001:]+result[6,10001:],c="k",linewidth="4",label="Pred")
plt.plot(result[0], result[1]+result[6],c="k",linewidth="4",label="Pred")
plt.plot( result[0], result[2],c="r",label="True")
plt.ylabel("State 0")
plt.subplot(3, 1, 2)
plt.plot(result[0,:10001], result[3,:10001],c="k",linewidth="4",label="Pred")
plt.plot(result[0,10001:], result[3,10001:]+result[7,10001:],c="k",linewidth="4",label="Pred")
plt.plot(result[0], result[4],c="r",label="True")
plt.ylabel("State 1")
plt.legend(loc="upper right")
plt.subplot(3, 1, 3)
plt.plot(result[0], result[5],c="k",label="Input")
plt.ylabel("Control Input")
plt.xlabel("Time")
plt.legend(loc="upper right")
plt.show()
plt.savefig("RNN10Systemid.svg")


#AutoCorrelation Residual Plot State-0

plt.clf()


residuals = ((result[2]-result[6]-result[1])[10001:])

crosscorr=NormCrossCorr(residuals.flatten(),residuals.flatten(),mode="same")
plt.plot(np.arange(-20000,20000),crosscorr,c="k",linewidth="4",label="AutoCorrelation-State0")
plt.ylabel("Correlation Value")
plt.xlabel("Lag")
plt.xlim(-1,1)
plt.legend()
plt.show()
plt.savefig("AutoCorrelationState0.svg")
plt.clf()

#QQPlot Residuals State-0
qqplot(residuals.flatten(),plot=plt)
plt.show()
plt.savefig("QQState0.svg")
plt.clf()

#AutoCorrelation Residual Plot State-1

residuals = ((result[4]-result[3]-result[7])[10001:])

crosscorr=NormCrossCorr(residuals.flatten(),residuals.flatten(),mode="same")
plt.plot(np.arange(-20000,20000),crosscorr,c="k",linewidth="4",label="AutoCorrelation-State1")
plt.ylabel("Correlation Value")
plt.xlabel("Lag")
plt.xlim(-1,1)
plt.legend()
plt.show()
plt.savefig("AutoCorrelationState1.svg")
plt.clf()

#QQPlot Residuals State-1

qqplot(residuals.flatten(),plot=plt)
plt.show()
plt.savefig("QQState1.svg")
plt.clf()

#CrossCorrelation Check
crosscorr=NormCrossCorr(result[2,10001:]-result[6,10001:],result[1,10001:],mode="same")
plt.plot(np.arange(-20000,20000),crosscorr,c="k",linewidth="4",label="CrossCorrelation")
plt.ylabel("Correlation Value")
plt.xlabel("Lag")
plt.xlim(-1,1)
plt.legend()
plt.show()
plt.savefig("CrossCorrelationState0.svg")
plt.clf()


crosscorr=NormCrossCorr(result[4,10001:]-result[7,10001:],result[3,10001:],mode="same")
plt.plot(np.arange(-20000,20000),crosscorr,c="k",linewidth="4",label="CrossCorrelation")
plt.ylabel("Correlation Value")
plt.xlabel("Lag")
plt.xlim(-1,1)
plt.legend()
plt.show()
plt.savefig("CrossCorrelationState1.svg")
plt.clf()


print "Velocity Error:",np.sqrt(np.mean((result[6,10001:]+result[1,10001:]-result[2,10001:])**2))
print "Acceleration Error:",np.sqrt(np.mean((result[7,10001:]+result[3,10001:]-result[4,10001:])**2))


print "Velocity R2:",np.corrcoef(-result[6,10001:]+result[2,10001:],result[1,10001:])
print "Acceleration R2:",np.corrcoef(result[3,10001:],result[4,10001:]-result[7,10001:])


