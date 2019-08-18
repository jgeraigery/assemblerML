# Please change the root variable as required.
# train and test directories of the dataset need to be present in the root directory
# weights need to be present in a weights folder in the root directory

from Evaluate import *

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
import pickle, math,random

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

random.seed(1)
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
        noisedot[:,:]=np.random.normal(0.0, 1, 2)
        noise= np.zeros([1,2],dtype=float)
        noise[:,:]=np.random.normal(0.0, 1, 2)
        #print noisedot,noisedot.shape,self.stateDot.shape
        #self.stateDot+=noisedot.transpose()
        self.state += self.stateDot.transpose() * self.dT
        #self.state+=noise*self.dT
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


def densemodel(features=3):
        input = Input(batch_shape=(None,features))
        output = Dense(2,activation="linear",use_bias=False)(input)
        model = Model(inputs=input, outputs=output)
        model.compile(loss="mse", optimizer='adam', metrics=['accuracy'])
        return model

def ddmodel(features=3):
        input = Input(batch_shape=(None,features))
        x = Dense(10,activation="relu",use_bias=False)(input)
        x = Dense(10,activation="relu",use_bias=False)(x)
        output = Dense(2,activation="linear",use_bias=False)(x)
        model = Model(inputs=input, outputs=output)
        model.compile(loss="mse", optimizer='adam', metrics=['accuracy'])
        return model

def rnnmodel(window=3,features=5):
        input = Input(batch_shape=(None,window,features))
        x=SimpleRNN(10,activation="relu",return_sequences=True)(input)
        x=SimpleRNN(10,activation="relu",return_sequences=True)(x)
        x=Flatten()(x)
        x=Dense(10,activation="relu")(x)
        output=Dense(2,activation="linear")(x)
        model = Model(inputs=input, outputs=output)
        adam=keras.optimizers.Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False)
        model.compile(loss="mse", optimizer=adam, metrics=['accuracy'])
        return model


def arimamodel(window=3,features=5):
        input = Input(batch_shape=(None,window,features))
        output=SimpleRNN(2,activation="linear")(input)
        model = Model(inputs=input, outputs=output)
        adam=keras.optimizers.Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False)
        model.compile(loss="mse", optimizer=adam, metrics=['accuracy'])
        return model


m = Motor()
dT = .001
m.setTimeStep(dT)
model = densemodel()
model2 = arimamodel(features=5)
result = np.zeros([8,1])
printDuring=False

movingvalue=np.zeros((3,5))

X=[]
X2=[]
y=[]
error=[]

for i in np.arange(0, 100000):
    # StepWise control input function
    if ( np.mod(i,50) == 0 ):
        print "Loop-",i
        controlInput=0
        if (i%1000==0 and i!=0):
                controlInput = getControlInput()
	if i==10000:
		print "Loop:",i
		m.Km=1
		#m.b+=10*dT
		m.update()

    print controlInput
    stateTensor =(m.state)
    stateTensor = np.concatenate((stateTensor,(np.ones([1,1], dtype=float) * controlInput)), 1)
    outBar=m.step(controlInput)

    if i<10000:
        out=np.zeros((1,2))
   	X.append(stateTensor)
        y.append(outBar-stateTensor[:,0:2])
    elif i==10000:
        out=np.zeros((1,2))
        model.fit(np.asarray(X).reshape(10000,3),(np.asarray(y)).reshape(10000,2),epochs=50,batch_size=32)
    elif (i>10000 and i<30000):
    	out=model.predict(stateTensor)

	movingvalue[0:2,:]=movingvalue[1:3,:]
	movingvalue[2,:]=np.concatenate((out,stateTensor),1)
	movingvalue2=np.asarray(list(movingvalue))
   	X2.append(movingvalue2)
	error.append(-out+(outBar-stateTensor[:,0:2]))

    elif i==30000:
	print "i"
	print np.asarray(X2).shape
	print np.asarray(error).shape
    	out=model.predict(stateTensor)

	movingvalue[0:2,:]=movingvalue[1:3,:]
	movingvalue[2,:]=np.concatenate((out,stateTensor),1)
	movingvalue2=np.asarray(list(movingvalue))
   	X2.append(movingvalue2)
	error.append(-out+(outBar-stateTensor[:,0:2]))

        model2.fit(np.asarray(X2),(np.asarray(error)).reshape(20000,2),epochs=50,batch_size=32)
	adam=keras.optimizers.Adam(lr=0.00001, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False)
        model2.compile(loss="mse", optimizer=adam, metrics=['accuracy'])
    elif i>30000:
    	out=model.predict(stateTensor)

	movingvalue[0:2,:]=movingvalue[1:3,:]
	movingvalue[2,:]=np.concatenate((out,stateTensor),1)
	movingvalue2=np.asarray(list(movingvalue))

	outerror=model2.predict(movingvalue2.reshape(1,3,5))
	outerror2=-out+(outBar-stateTensor[:,0:2])

        #model2.fit(np.concatenate((out,stateTensor),1),outerror2,epochs=1)
	out+=outerror
    else:
	continue
    tmpResult = np.empty([8,1])
    tmpResult[0] = dT*(i+1)
    tmpResult[1] = out[0][0]
    tmpResult[2] = outBar[0][0]
    tmpResult[3] = out[0][1]
    tmpResult[4] = outBar[0][1]
    tmpResult[5] = controlInput
    tmpResult[6] = stateTensor[:,0]
    tmpResult[7] = stateTensor[:,1]
    result = np.concatenate((result,tmpResult),1)


evaluate(result,"StateSpace-dd")

print "Weights",model.get_weights()
print "A and B",m.A,m.B

print m.Km
