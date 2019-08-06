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
from keras.layers import Convolution2D, Dense, Input, MaxPooling2D, Dropout, Flatten, ZeroPadding2D, Activation,LSTM,Bidirectional,Convolution1D,MaxPooling1D,Conv1D,SimpleRNN,Lambda,BatchNormalization
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

from nalu import NALU
from nac import NAC

def RNNNALUModel(time_step=1,output_time_step=1,input_size=3,output_size=2,lr=0.0001,width=10,depth=1,batchnorm=False):
	input = Input(batch_shape=(None,time_step,input_size))
	inputnew = input
	if batchnorm:
		inputnew=BatchNormalization()(inputnew)
	if depth==1: #Checking for depth to set return_sequences Flag
		x=SimpleRNN(width,activation="relu",return_sequences=False)(inputnew)

	elif depth>1:
		x=SimpleRNN(width,activation="relu",return_sequences=True)(inputnew)
		for i in range(depth-2):
			x=SimpleRNN(width,activation="relu",return_sequences=True)(x)

		x=SimpleRNN(width,activation="relu",return_sequences=False)(x) #Last Layer needs to have return_sequences flag as False


	output = NALU(output_time_step*output_size,use_gating=True,activation="linear")(x) #Dense Layer of Size Number of Time Steps * Number of Features or States
	output=Lambda(lambda xin :K.reshape(xin,(-1,output_time_step,output_size)))(output) #Reshaping

	model = Model(inputs=input, outputs=output) #Creating Model
	adam=keras.optimizers.Adam(lr=lr, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False) #Optimizer
	model.compile(loss="mse", optimizer=adam, metrics=['accuracy','mse']) #Compiling Model

	return model


