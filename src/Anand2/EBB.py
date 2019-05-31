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
from keras.layers import Convolution2D, Dense, Input, MaxPooling2D, Dropout, Flatten, ZeroPadding2D, Activation,LSTM,Bidirectional,Convolution1D,MaxPooling1D,Conv1D,SimpleRNN,Lambda,Conv2D,BatchNormalization
from keras.layers.pooling import GlobalAveragePooling2D,GlobalAveragePooling1D
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
from batch_renorm import BatchRenormalization as BRN

seed=10
np.random.seed(seed)


X=np.load("CK.npy")
y=np.load("Em.npy")
AU=np.load("OneHotAU.npy")

y=keras.utils.to_categorical(y, num_classes=8)

print X.shape
print y.shape

def black_box():
        input = Input(batch_shape=(None,26))       
        x=Dense(100,activation="relu")(input)
        x=Dense(100,activation="relu")(x)
        output = Dense(8,activation="softmax")(x)
        model = Model(inputs=input, outputs=output)     
       
	return model  


def extractor_model():
	input=Input(batch_shape=(None,500,500,3))
	x=BRN(axis=-1, momentum=0.3, epsilon=1e-5)(input) 
	x=Conv2D(32, kernel_size=(3, 3), strides=(1, 1),activation='relu')(x)
	x=Conv2D(32, kernel_size=(3, 3), strides=(1, 1),activation='relu')(x)
	x=MaxPooling2D(pool_size=(2, 2), strides=(2, 2))(x)
	x=Conv2D(64, kernel_size=(3, 3), strides=(1, 1),activation='relu')(x)
	x=Conv2D(64, kernel_size=(3, 3), strides=(1, 1),activation='relu')(x)
	x=BRN(axis=-1, momentum=0.3, epsilon=1e-5)(x) 
	x=MaxPooling2D(pool_size=(2, 2), strides=(2, 2))(x)
    
	x = GlobalAveragePooling2D()(x)                                                                                                                              
	#x=BatchNormalization()(x)                                                                                                             
	x = Dense(100, activation='relu')(x)                                                                                                                                  
	x = Dense(100, activation='relu')(x)
	x=BRN(axis=-1, momentum=0.3, epsilon=1e-5)(x) 
	#x=BatchNormalization()(x)                                                                                                             
	output = Dense(26,activation="sigmoid")(x)                                                                   
                          
	model = Model(inputs=input, outputs=output)
	model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

	return model

extractor=extractor_model()
blackbox=black_box()

blackbox.load_weights("weights/BlackBox.hdf5")
blackbox.trainable=False

for layer in blackbox.layers:
        layer.trainable=False

for layer in blackbox.layers:
        print layer.name, layer.trainable


blackboxout=blackbox((extractor.output))
model=Model(extractor.input,blackboxout)

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

print model.summary()

layer=model.layers[-1]
print layer.name, layer.trainable

checkpoint=ModelCheckpoint("weights/ExtractorFrozen.hdf5",monitor="loss",verbose=1,save_best_only=True)

#extractor.fit(X[0:100], AU[0:100], epochs=50,verbose=1)

model.fit(X, y, epochs=00,verbose=1,callbacks=[checkpoint])

model.load_weights("weights/ExtractorFrozen.hdf5")

pred, pred_acc= model.evaluate(X,y)

print (pred)
print pred_acc
'''
print model.predict(X)
print extractor.predict(X)
print blackbox.predict(extractor.predict(X))


y=extractor.predict(X)
np.save("extractor_pred.npy",y)
y=model.predict(X)
np.save("blackbox_pred.npy",y)

print np.argmax(y,axis=1)
'''
