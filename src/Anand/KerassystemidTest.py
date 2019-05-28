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
from sklearn.metrics import mean_squared_error as mse

np.random.seed(10)

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


def getControlInput():
    out = np.random.rand() * 20 - 10
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
        input = Input(batch_shape=(None,1,3))
        x=SimpleRNN(10,activation="relu",return_sequences=True)(input)
        output = Dense(2)(x)
        model = Model(inputs=input, outputs=output)
        model.compile(loss="mse", optimizer='adam', metrics=['accuracy'])
        return model


m = Motor()
dT = .001
m.setTimeStep(dT)
model = densemodel()
X=np.load("X20000.npy")
y=np.load("y20000.npy")
model.fit(X.reshape(X.shape[0],3),y.reshape(20000,2),epochs=20)
result = np.zeros([6,1])
printDuring=False
controlact=np.load("RNNresult.npy")
controlact=controlact[5]
for i in np.arange(0, 50000):
    #print "Loop-",i
    # control input function
    if ( np.mod(i,50) == 0 ):
        controlInput = controlact[i+1]


    stateTensor =(m.state)
    stateTensor = np.concatenate((stateTensor,(np.ones([1,1], dtype=float) * controlInput)), 1)
    outBar=m.step(controlInput)
    outBar = (outBar)
    out=model.predict(stateTensor)
    #model.fit(stateTensor,outBar,epochs=1)
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

plt.figure(2)
plt.subplot(2, 1, 1)
plt.plot(result[0], result[1],c="k",label="Pred")
plt.plot( result[0], result[2],c="r",label="True")
plt.ylabel("State 0")
plt.subplot(2, 1, 2)
plt.plot(result[0], result[3],c="k",label="Pred")
plt.plot(result[0], result[4],c="r",label="True")
plt.ylabel("State 1")
plt.xlabel("Time")
plt.legend()
plt.show()
plt.savefig("DenseSystemidtest.svg")

np.save("Denseresult.npy",result)

print "Velocity Error:",np.sqrt(np.mean((result[1]-result[2])**2))
print "Acceleration Error:",np.sqrt(np.mean((result[3]-result[4])**2))

