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

def BoostingModel(model1,model2,lr=0.001,time_step=1,input_size=3,output_size=2): 	#If Boosting a Booster Model1 must be the  Boosted Model OtherWise errors

											#Checking if Model2 is Boosted Model and Raising Error
	if type(model2.output_shape) is list:
		raise ValueError( "Model-2 cannot be a Boosted Model Please make Model-1 as the Boosted Model")
		return -1

	input = Input(batch_shape=(None,time_step,input_size))
	#action=crop(2,input_size-1,input_size)(input) 					#Getting Input Action


											#Getting Model1 Prediction

	if type(model1.output_shape) is list: 						#Finding if Model1 is Boosted Model
		output_a_1,output_a_2 = model1(input)
		output_a=keras.layers.add([output_a_1,output_a_2]) 			#Adding Outputs together to get Boosted Model's Final Prediction 
	else: 										#If Model1 is not Boosted

		if len(model1.input_shape)==3:
			output_a= model1(input)
			
											#Getting Model2 Prediction
	#input_b=keras.layers.concatenate([output_a,action])
	input_b=BatchNormalization()(output_a)
	output_b = model2(input_b)

	model = Model(inputs=input, outputs=[output_a,output_b])					#Compiling Models together

	adam=keras.optimizers.Adam(lr=lr, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False)
	model.compile(loss="mse", optimizer=adam, metrics=['accuracy'])
	return model
