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


def ControlModel(model1,model2,lr=0.001,time_step=1,input_size=3,output_size=2): 	#If Boosting a Booster Model1 must be the  Boosted Model OtherWise errors

	input1 = Input(batch_shape=(None,time_step,input_size))				#Current Position
	input2=crop(2,0,4)(input1) 											#Getting Model2 Prediction

											#Getting Model1 Prediction
	output_a= model1([input1])

	input3=keras.layers.concatenate([input2,output_a],axis=-1)

	output_b = model2(input3)

	model = Model(inputs=[input1], outputs=[output_a,output_b])					#Compiling Models together

	adam=keras.optimizers.Adam(lr=lr, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False)
	model.compile(loss="mse", optimizer=adam, metrics=['accuracy'])
	return model

