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
from keras.layers import Convolution2D, Dense, Input, MaxPooling2D, Dropout, Flatten, ZeroPadding2D, Activation,LSTM,Bidirectional,Convolution1D,MaxPooling1D,Conv1D,SimpleRNN,Lambda
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



def DenseModel(input_size=3,time_step=1,output_time_step=1,output_size=2,lr=0.0001,width=10,depth=1):
	input = Input(batch_shape=(None,time_step,input_size))
	inputnew=Lambda(lambda xin :K.reshape(xin,(-1,time_step*input_size)))(input)
	x = Dense(width,activation="relu",use_bias=False)(inputnew)

	for i in range(depth-1):
		x = Dense(width,activation="relu",use_bias=False)(x)

	output = Dense(output_time_step*output_size,activation="linear",use_bias=False)(x)
	output=Lambda(lambda xin :K.reshape(xin,(-1,output_time_step,output_size)))(output)
	
	model = Model(inputs=input, outputs=output)
	adam=keras.optimizers.Adam(lr=lr, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False)

	model.compile(loss="mse", optimizer=adam, metrics=['accuracy'])

	return model
