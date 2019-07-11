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
from keras.layers import Convolution2D, Dense, Input, MaxPooling2D, Dropout, Flatten, ZeroPadding2D, Activation,LSTM,Bidirectional,Convolution1D,MaxPooling1D,Conv1D,SimpleRNN,Lambda,Reshape,TimeDistributed
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



def EnsembleModel(model1,model2,lr=0.001,time_step=1,input_size=3,output_size=2):
	input = Input(batch_shape=(None,time_step,input_size))
										#Finding Model1 Prediction
	if type(model1.output_shape) is list:						#Checking if Model1 is Boosted
		output_a_1,output_a_2 = model1(input)
		output_a = keras.layers.add([output_a_1, output_a_2])			#Adding Boosted model's predictions together
	else:										#If Model1 is not Boosted
		output_a= model1(input)
											#Finding Model2 Prediction
	if type(model2.output_shape) is list:
		output_b_1,output_b_2 = model2(input)
		output_b = keras.layers.add([output_b_1, output_b_2])
	else:
		output_b= model2(input)


	output = keras.layers.concatenate([output_a, output_b],axis=2)			#Concatenating to weigh Model1 and Model2
	output=TimeDistributed(Dense(output_size))(output)					#Weighted Average
	model = Model(inputs=input, outputs=output)					#Combining Models
	adam=keras.optimizers.Adam(lr=lr, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False)
	model.compile(loss="mse", optimizer=adam, metrics=['accuracy'])
	return model
