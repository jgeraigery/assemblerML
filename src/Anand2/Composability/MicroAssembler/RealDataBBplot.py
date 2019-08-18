# Please change the root variable as required.
# train and test directories of the dataset need to be present in the root directory
# weights need to be present in a weights folder in the root directory

from Problems.DCMotor import *
from numpy import sin,cos,pi
from Models.Dense import* 
from Models.NALU import* 
from Models.RNNNALU import* 
from Models.StateSpace import* 
from Models.RNN import* 
from Operators.Ensemble import* 
from Operators.Boosting import* 
from Evaluation.Evaluate import* 
from Operators.nalu import NALU
from Operators.nac import NAC
from scipy.stats import spearmanr

from matplotlib.animation import FuncAnimation,FFMpegFileWriter


matplotlib.rcParams['animation.ffmpeg_args'] = '-report'
matplotlib.rcParams['animation.bitrate'] = 2000


import pandas as pd
#Name="10Input10Output-Sphere-RNNBoostedSS"

Name="1Input1Output-Sphere-RNNBoostedSS"

time_step=1
output_time_step=1

input_size=10
output_size=3


#model1 = RNNModel(time_step=time_step,output_time_step=output_time_step,input_size=input_size,output_size=output_size,lr=0.001,depth=10,width=100)
model1 = SSModel(time_step=time_step,output_time_step=output_time_step,input_size=input_size,output_size=output_size,lr=0.001)
model2 = RNNModel(time_step=output_time_step,output_time_step=output_time_step,input_size=output_size,output_size=output_size,lr=0.001)



X=np.load("1Input1OutputSphereExtrapolationX2.npy")
#X1=np.load("1Input1OutputX3.npy")
y=np.load("1Input1OutputSphereExtrapolationy2.npy")



'''
X2=np.zeros((len(X)-20,10,10))
y=np.zeros((len(X)-20,10,3))
for i in range(10,len(X)-10):
	X2[i-10,:,:]=np.array(np.reshape(X[i-10:i],(1,10,10)))
        y[i-10,:,:]=np.array(np.reshape(X[i:i+10,:,0:3]-X[i-1:i+9,:,0:3],(1,10,3)))


X=np.array(X2)


X2=np.zeros((len(X1)-20,10,10))
y1=np.zeros((len(X1)-20,10,3))
for i in range(10,len(X1)-10):
	X2[i-10,:,:]=np.array(np.reshape(X1[i-10:i],(1,10,10)))
        y1[i-10,:,:]=np.array(np.reshape(X1[i:i+10,:,0:3]-X1[i-1:i+9,:,0:3],(1,10,3)))


X1=np.array(X2)

X=np.concatenate((X,X1),axis=0)
y=np.concatenate((y,y1),axis=0)
'''

reduce_lr = ReduceLROnPlateau(monitor='loss', factor=0.5,
                              patience=10, min_lr=0.00001,verbose=1)

early_stop=EarlyStopping(monitor='loss', min_delta=0.000005, patience=50, verbose=1)
#model1.fit(X,y,epochs=1000,batch_size=512,callbacks=[reduce_lr,early_stop])

pred=model1.predict(X)

model1.trainable=False

model = BoostingModel(model1,model2,time_step=time_step,input_size=input_size,output_size=output_size,lr=0.001)

print (model.summary())


#model.fit(X,[y,y-pred],epochs=1000,batch_size=512,callbacks=[reduce_lr,early_stop])


#model.save_weights(Name+".hdf5")
model.load_weights(Name+".hdf5")

#Name="10Input10Output-Chiplets-RNNBoostedSS-W-30-D-10"


out1,out2=model.predict(X)

out=out1+out2

#out=model1.predict(X)
for j in range(1):
	for i in range(3):
		print(" Deep Model "+Name+" RMSE Error of State:",j,i,np.sqrt(np.mean((out[:,j,i]-y[:,j,i])**2)))
		print("Alpha 0.5 R2 of State:",i,np.corrcoef(out[:,j,i].T,y[:,j,i].T))
		#print("Spearmanr of State:",i,spearmanr(out[:,0,i],y[:,0,i]))



plt.figure(1,figsize=(20,10))
plt.subplot(2, 1,1)
plt.title("State0 BoxPlot Error")
plt.boxplot(np.sqrt((out[:,:,0]-y[:,:,0])**2))
plt.subplot(2, 1, 2)
plt.title("State1 BoxPlot Error")
plt.boxplot(np.sqrt((out[:,:,1]-y[:,:,1])**2))
plt.ylim(0,50)
plt.show()
plt.savefig(Name+"BxPlot.jpg")
plt.clf()


