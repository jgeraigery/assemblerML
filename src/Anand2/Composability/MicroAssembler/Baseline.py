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

DataFrame=pd.read_csv("particles - 3.tsv",sep='\t',header=0)

print(DataFrame.columns)

DFT=DataFrame[DataFrame['mode']=="Tracking"]

DFT.drop(columns=["mode","ctrl.img.filename","id","type","frame"])

DataFrame=DFT

time=np.asarray(DataFrame["t, s"])

particle=np.asarray(DataFrame[["Px","Py","Pt"]])
sprite=np.asarray(DataFrame[["Sx","Sy","St"]])
target=np.asarray(DataFrame[["Tx","Ty","Tt"]])

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
	if np.abs((time[i+1]-time[i])-0.017)>0.005:
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

		angleRad=outBar[2]*np.pi/180
		alphaRad=(outBar[2]+90)*np.pi/180

		if(np.cos(alphaRad) < 0.001):
			maxDist=0.001
		else:
			maxDist=np.abs(np.cos(alphaRad))*np.abs(np.tan(alphaRad)*stateTensor[0]-stateTensor[1]-tan(alphaRad)*outBar[0]+outBar[1])
		
		moving_pred[output_time_step-1,0]=stateTensor[4]-(stateTensor[0]+min(maxDist,0.001)*np.cos(angleRad))
		moving_pred[output_time_step-1,1]=stateTensor[5]-(stateTensor[1]+min(maxDist,0.001)*np.sin(angleRad))
		moving_pred[output_time_step-1,2]=(outBar[2]-stateTensor[2])
		moving_pred2=np.asarray(list(moving_pred))
	
		X.append(moving_input2)
		y.append(moving_output2)
		out.append(moving_pred2)

X=np.asarray(X)
y=np.asarray(y)
out=np.asarray(out)

for i in range(3):
	print("Alpha 0.5 RMSE Error of State:",i,np.sqrt(np.mean((out[:,:,i]-y[:,:,i])**2)))
	print("Alpha 0.5 R2 of State:",i,np.corrcoef(out[:,:,i].T,y[:,:,i].T))
	print("Spearmanr of State:",i,spearmanr(out[:,:,i],y[:,:,i]))

