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
        #print(self.stateDot)
        self.stateDot+=noisedot.transpose()
        self.state += self.stateDot.transpose() * self.dT
        self.state+=noise*self.dT
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

def rnnmodel():
        input = Input(batch_shape=(None,3,3))
        x=SimpleRNN(10,activation="relu",return_sequences=True)(input)
        x=SimpleRNN(10,activation="relu",return_sequences=True)(x)
        x=Flatten()(x)
        x=Dense(10,activation="relu")(x)
        output=Dense(2,activation="linear")(x)
        model = Model(inputs=input, outputs=output)
	adam=keras.optimizers.Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False)
        model.compile(loss="mse", optimizer=adam, metrics=['accuracy'])
        return model


m = Motor()
dT = .001
m.setTimeStep(dT)
model = rnnmodel()
print model.summary()
result = np.zeros([8,1])
printDuring=False

X=[]
y=[]
movingvalue=np.zeros((3,3))

for i in np.arange(0, 50000):
    # control input function
    if ( np.mod(i,50) == 0 ):
        print ("Loop-",i)
	controlInput=0
	if i%1000==0 and i!=0:
        	controlInput = 10
        	#controlInput = getControlInput()
    if (i%10000==0):
	#m.J=m.J+0.1
	m.update()
    stateTensor =(m.state)
    stateTensor = np.concatenate((stateTensor,(np.ones([1,1], dtype=float) * controlInput)), 1)
    outBar=m.step(controlInput)
    movingvalue[0:2,:]=movingvalue[1:3,:]
    movingvalue[2,:]=stateTensor
    movingvalue2=np.asarray(list(movingvalue))

    if i<10000:
        out=np.zeros((1,2))
        X.append(movingvalue2)
        y.append(outBar-stateTensor[:,0:2])

    elif i==10000:
        out=np.zeros((1,2))
        model.fit(np.asarray(X),(np.asarray(y)).reshape(10000,2),epochs=50)
	adam=keras.optimizers.Adam(lr=0.00001, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False)
        model.compile(loss="mse", optimizer=adam, metrics=['accuracy'])
    elif i>10000:
    	out=model.predict(movingvalue2.reshape(1,3,3))
    	model.fit(movingvalue2.reshape(1,3,3),outBar-stateTensor[:,0:2],epochs=1)
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

np.save("RNN10results.npy",result)
evaluate(result,"RNNArima")
