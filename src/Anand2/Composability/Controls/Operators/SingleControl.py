# Please change the root variable as required.
# train and test directories of the dataset need to be present in the root directory
# weights need to be present in a weights folder in the root directory

from keras import applications
from keras.models import Sequential
from keras.layers.core import Flatten, Dense, Dropout
from keras.layers.convolutional import Convolution2D, MaxPooling2D, ZeroPadding2D
from keras.optimizers import SGD

import keras
from keras.layers.advanced_activations import LeakyReLU, PReLU,ELU
from keras.preprocessing.image import ImageDataGenerator
from keras.layers import Convolution2D, Dense, Input, MaxPooling2D, Dropout, Flatten, ZeroPadding2D, Activation,LSTM,Bidirectional,Convolution1D,MaxPooling1D,Conv1D,SimpleRNN,Lambda,Reshape,BatchNormalization
from keras.layers.pooling import AveragePooling2D,GlobalAveragePooling1D
from keras.models import Model, Sequential
from keras.utils.vis_utils import plot_model
from keras.optimizers import SGD, Adam
from keras.regularizers import l2
from keras.callbacks import ModelCheckpoint, LearningRateScheduler, ReduceLROnPlateau,EarlyStopping
import keras.backend as K
from keras.preprocessing.image import ImageDataGenerator
from keras import regularizers
from keras.metrics import categorical_accuracy
from keras.utils import np_utils

def crop(dimension, start, end):
	# Crops (or slices) a Tensor on a given dimension from start to end
	# example : to crop tensor x[:, :, 5:10]
	# call slice(2, 5, 10) as you want to crop on the second dimension
	def func(x):
		if dimension == 0:
			return x[start: end]
		if dimension == 1:
			return x[:, start: end]
		if dimension == 2:
			return x[:, :, start: end]
		if dimension == 3:
			return x[:, :, :, start: end]
		if dimension == 4:
			return x[:, :, :, :, start: end]

	return Lambda(func)

def customLoss(input1,input2):
	def loss(y_true,y_pred):
		return (K.square(input2-y_pred)+10*K.square(y_true-y_pred))
	return loss

def customLoss2(y_true,y_pred):
	return (K.square(y_true-y_pred)+K.square(y_true-y_pred))


def SingleControlModel(model1,lr=0.0001,time_step=1,input_size=2,output_size=1): 	#Model1 Controller, Model2 System ID, Model3 Inverse of Controller
	input1 = Input(batch_shape=(None,time_step,input_size))				#Xt Current Position
	input2 = Input(batch_shape=(None,time_step,input_size))				#Reference Position
	input3 = Input(batch_shape=(None,time_step,1))					#Ut Previous Action
	input4=keras.layers.concatenate([input1,input2,input3])				#Concatenating Xt,et,Ut to predict Ut+1
	output_a1,output_a2= model1(input4)							#Predicting Ut+1

	model2 = Model(inputs=[input1,input2,input3], outputs=[output_a1,output_a2])		#Compiling Models together but without output_a

	adam=keras.optimizers.Adam(lr=lr, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False)
	model2.compile(loss=[customLoss(input1,input2),customLoss2], optimizer=adam, metrics=['accuracy'])
	return model2
