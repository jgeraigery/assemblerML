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
from keras.callbacks import ModelCheckpoint, LearningRateScheduler, ReduceLROnPlateau,EarlyStopping,TensorBoard
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

def custom_loss(y_true, y_pred):
	y_true1=crop(2,0,1)(y_true)
	y_pred1=crop(2,0,1)(y_pred)
	y_true2=crop(2,1,2)(y_true)
	y_pred2=crop(2,1,2)(y_pred)
	return (1000*(y_true1-y_pred1)**2+1*(y_true2-y_pred2)**2)

def CyclicControlModel(model1,model2,model3,lr=0.0001,time_step=1,input_size=2,output_size=1): 	#Model1 Controller, Model2 System ID, Model3 Inverse of Controller

	model2.trainable=False

	input1 = Input(batch_shape=(None,time_step,input_size))				#Xt Current Position
	input2 = Input(batch_shape=(None,time_step,input_size))				#et error with Ref
	input3 = Input(batch_shape=(None,time_step,1))					#Ut Previous Action
	input4=keras.layers.concatenate([input1,input2,input3])				#Concatenating Xt,et,Ut to predict Ut+1
	#input4=keras.layers.concatenate([input1,input2])				#Concatenating Xt,et,Ut to predict Ut+1
	output_a= model1(input4)							#Predicting Ut+1

	input5=keras.layers.concatenate([input1,output_a])				#Concatenating Xt with Ut+1

	if type(model2.output_shape) is list: 						#Finding if SystemID is Boosted Model
		output_b_1,output_b_2 = model2(input5)
		output_b=keras.layers.add([output_b_1,output_b_2]) 			#Adding Outputs together to get Boosted Model's Final Prediction 
	else: 										#If Model2 is not Boosted

		output_b= model2(input5)						#Getting Next TD Position From SystemID using Xt,Ut+1
			
	#output_b=keras.layers.add([input1,output_b])					#Adding TD to Previous State to get New Position Xt+1
	#output_b2=crop(2,0,1)(output_b)							#Adding TD to Previous State to get New Position Xt+1
	input7=keras.layers.concatenate([output_b,output_a])				#Concatenating Xt+1 and Ut+1 to predict Xt

	#output_c = model3(input7)							#Get Xt

	#model4 = Model(inputs=[input1,input2,input3], outputs=[output_b,output_c])		#Compiling Models together but without output_a
	model4 = Model(inputs=[input1,input2,input3], outputs=[output_b])			#Compiling Models together but without output_a
	#model5 = Model(inputs=[input1,input2,input3], outputs=[output_a,output_b,output_c])	#Compiling Models together
	model5 = Model(inputs=[input1,input2,input3], outputs=[output_a,output_b])		#Compiling Models together

	adam=keras.optimizers.Adam(lr=lr, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False)
	model4.compile(loss=custom_loss, optimizer=adam, metrics=['accuracy'])
	model5.compile(loss=custom_loss, optimizer=adam, metrics=['accuracy'])
	return model4,model5
