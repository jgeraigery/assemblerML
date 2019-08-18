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
Name="10Input1OutputDense-W-512-D-10"

DataFrame=pd.read_csv("SphereparticlesExtrapolation - 2.tsv",sep='\t',header=0)

print(DataFrame.columns)

DFT=DataFrame[DataFrame['mode']=="Tracking"]

DFT.drop(columns=["mode","ctrl.img.filename","id","type","frame"])

DataFrame=DFT

time=np.asarray(DataFrame["t, s"])

particle=np.asarray(DataFrame[["Px","Py","Pt"]])
sprite=np.asarray(DataFrame[["Sx","Sy","St"]])
target=np.asarray(DataFrame[["Tx","Ty","Tt"]])
predicted=np.asarray(DataFrame[["Ex","Ey","Et"]])


for i in range(3):
	print("Alpha 0.5 RMSE Error of State:",i,np.sqrt(np.mean((predicted[:,i]-particle[:,i])**2)))
	print("Alpha 0.5 R2 of State:",i,np.corrcoef(predicted[:,i].T,particle[:,i].T))
	print("Spearmanr of State:",i,spearmanr(predicted[:,i],particle[:,i]))


time_step=1
output_time_step=1

input_size=10
output_size=3

X=[]
y=[]
out=[]
moving_input=np.zeros((time_step,input_size))
moving_output=np.zeros((output_time_step,output_size))
moving_pred=np.zeros((output_time_step,output_size))

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


		if output_time_step>1:
			moving_pred[0:output_time_step-1,:]=moving_pred[1:output_time_step,:]

		moving_pred[output_time_step-1,:]=predicted[i+1]-stateTensor[0:3]
		moving_pred2=np.asarray(list(moving_pred))
	
		X.append(moving_input2)
		y.append(moving_output2)
		out.append(moving_pred2)

X=np.asarray(X)

y=np.asarray(y)
out=np.asarray(out)

ynew=y[4:len(y),:,:]
outnew=out[0:len(y)-4,:,:]

y=ynew
out=outnew

for i in range(3):
	print("Alpha 0.5 RMSE Error of State:",i,np.sqrt(np.mean((y[:,:,i]-out[:,:,i])**2)))
	print("Alpha 0.5 R2 of State:",i,np.corrcoef(out[:,:,i].T,y[:,:,i].T))
	print("Spearmanr of State:",i,spearmanr(out[:,:,i],y[:,:,i]))
