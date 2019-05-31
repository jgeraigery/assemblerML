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

from nalu import NALU
#from nac import NAC

seed=10
np.random.seed(seed)

from scipy.special import expit

anger=np.zeros((26,1000))
contempt=np.zeros((26,1000))
disgust=np.zeros((26,1000))
fear=np.zeros((26,1000))
happy=np.zeros((26,1000))
sad=np.zeros((26,1000))
surprise=np.zeros((26,1000))
neutral=np.zeros((26,1000))

anger[3,:]=1
anger[6,:]=1
anger[9,:]=np.random.choice(a=[1, 0], size=(1000), p=[0.26, 1-0.26])
anger[16,:]=np.random.choice(a=[1, 0], size=(1000), p=[0.52, 1-0.52])
anger[22,:]=np.random.choice(a=[1, 0], size=(1000), p=[0.29, 1-0.29])
anger[23,:]=1

contempt[3,:]=1
contempt[6,:]=1
contempt[9,:]=1
contempt[11,:]=1
contempt[12,:]=1

disgust[3,:]=np.random.choice(a=[1, 0], size=(1000), p=[0.31, 1-0.31])
disgust[8,:]=1
disgust[9,:]=1
disgust[16,:]=1
disgust[23,:]=np.random.choice(a=[1, 0], size=(1000), p=[0.26, 1-0.26])

fear[0,:]=1
fear[1,:]=np.random.choice(a=[1, 0], size=(1000), p=[0.57, 1-0.57])
fear[3,:]=1
fear[4,:]=np.random.choice(a=[1, 0], size=(1000), p=[0.63, 1-0.63])
fear[19,:]=1
fear[24,:]=1
fear[25,:]=np.random.choice(a=[1, 0], size=(1000), p=[0.33, 1-0.33])

happy[5,:]=np.random.choice(a=[1, 0], size=(1000), p=[0.51, 1-0.51])
happy[11,:]=1
happy[24,:]=1

sad[0,:]=np.random.choice(a=[1, 0], size=(1000), p=[0.60, 1-0.60])
sad[3,:]=1
sad[14,:]=1
sad[10,:]=np.random.choice(a=[1, 0], size=(1000), p=[0.26, 1-0.26])
sad[16,:]=np.random.choice(a=[1, 0], size=(1000), p=[0.67, 1-0.67])


surprise[0,:]=1
surprise[1,:]=1
surprise[4,:]=np.random.choice(a=[1, 0], size=(1000), p=[0.66, 1-0.66])
surprise[24,:]=1
surprise[25,:]=1

X=np.concatenate((neutral,anger,contempt,disgust,fear,happy,sad,surprise),axis=1)
X=np.swapaxes(X,0,1)
y=np.asarray([0,1,2,3,4,5,6,7])
y=np.repeat(y,1000) 
 

print X[0],X[1000],X[2000]
print y[0],y[1000],y[2000]
X_train, X_test, y_train, y_test=train_test_split(X, y, test_size=0.1, random_state=42)

print X_train
print y_train
print np.bincount(y_train)
def make_model():
	input = Input(batch_shape=(None,26))
	#x=keras.layers.GaussianNoise(0.1)(input)
	x=Dense(100,activation="relu")(input)
	x=Dense(100,activation="relu")(x)
	output = Dense(8,activation="softmax")(x)
	model = Model(inputs=input, outputs=output)
	model.compile(loss="sparse_categorical_crossentropy", optimizer='adam', metrics=['accuracy'])
	#model.compile(loss="mse", optimizer='adam', metrics=['accuracy'])

	return model

model = make_model()
print model.summary()
checkpoint= ModelCheckpoint("weights/BlackBox.hdf5",monitor="loss",save_best_only=True)
model.fit(X_train, y_train, batch_size=256, epochs=200, verbose=1,callbacks=[checkpoint])
#pred, pred_acc= model.evaluate(X_train,y_train)
pred, pred_acc= model.evaluate(X_test,y_test)


print pred_acc
model.load_weights("weights/BlackBox.hdf5")
print model.predict(X)
