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

import pandas as pd
Name="1Input1OutputRNNBoostedSS-W-30-D-10"

DataFrame=pd.read_csv("particles - 3.tsv",sep='\t',header=0)

print(DataFrame.columns)

DFT=DataFrame[DataFrame['mode']=="Tracking"]

DFT.drop(columns=["mode","ctrl.img.filename","id","type","frame"])

DataFrame=DFT

time=np.asarray(DataFrame["t, s"])

particle=np.asarray(DataFrame[["Px","Py","Pt"]])
sprite=np.asarray(DataFrame[["Sx","Sy","St"]])
target=np.asarray(DataFrame[["Tx","Ty","Tt"]])

time_step=10
output_time_step=1

input_size=10
output_size=3


model1 = SSModel(time_step=time_step,output_time_step=output_time_step,input_size=input_size,output_size=output_size,lr=0.001)
model2 = RNNModel(time_step=output_time_step,output_time_step=output_time_step,input_size=output_size,output_size=output_size,lr=0.001,depth=3,width=10)


X=[]
y=[]

moving_input=np.zeros((time_step,input_size))
moving_output=np.zeros((output_time_step,output_size))

for i in np.arange(0,len(time)-1):
    # control input function
	#if np.abs((time[i+1]-time[i])-0.017)>0.005:
	if False:
		continue
	else:
		stateTensor=np.append(particle[i],sprite[i])
		stateTensor=np.append(stateTensor,target[i])
		stateTensor=np.append(stateTensor,time[i+1]-time[i])
		outBar=particle[i+1]

		if time_step>1: #To shift Inputs to the left
			moving_input[0:time_step-1,:]=moving_input[1:time_step,:]
	
		moving_input[time_step-1,:]=stateTensor
		moving_input2=np.asarray(list(moving_input))
	
		if output_time_step>1:
			moving_output[0:output_time_step-1,:]=moving_output[1:output_time_step,:]
	
		moving_output[output_time_step-1,:]=outBar-stateTensor[0:3]
		moving_output2=np.asarray(list(moving_output))
	
		X.append(moving_input2)
		y.append(moving_output2)

X=np.asarray(X)
y=np.asarray(y)


reduce_lr = ReduceLROnPlateau(monitor='loss', factor=0.5,
                              patience=8, min_lr=0.00001,verbose=1)

early_stop=EarlyStopping(monitor='loss', min_delta=0, patience=20, verbose=1)
model1.fit(X,y,epochs=500,batch_size=512,callbacks=[reduce_lr,early_stop])

pred=model1.predict(X)

model1.trainabale=False

model = BoostingModel(model1,model2,time_step=time_step,input_size=input_size,output_size=output_size,lr=0.001)

print (model.summary())


model.fit(X,[y,y-pred],epochs=500,batch_size=512,callbacks=[reduce_lr,early_stop])


model.save_weights(Name+".hdf5")
#model.load_weights(Name+".hdf5")

out1,out2=model.predict(X)

out=out1+out2

for i in range(3):
	print(" Deep Model "+Name+" RMSE Error of State:",i,np.sqrt(np.mean((out[:,:,i]-y[:,:,i])**2)))
	print("Alpha 0.5 R2 of State:",i,np.corrcoef(out[:,:,i].T,y[:,:,i].T))
	print("Spearmanr of State:",i,spearmanr(out[:,:,i],y[:,:,i]))




result=np.concatenate((out[:,[-1],:],y[:,[-1],:]+X[:,[-1],0:output_size],X[:,[-1],0:output_size]),axis=2)

result=result.reshape(len(X),output_size*3)

time=time.reshape(len(time),1)
time=time[:len(time)-1]

result=np.concatenate((result,time),axis=1)
evaluate(result,output_size=output_size,Training_Time=0,name="Images/TrainData"+Name)

