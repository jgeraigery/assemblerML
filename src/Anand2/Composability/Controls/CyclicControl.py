# Please change the root variable as required.
# train and test directories of the dataset need to be present in the root directory
# weights need to be present in a weights folder in the root directory

from Problems.DCMotor import *
from numpy import sin,cos,pi
from Models.ControlDense import* 
from Models.Dense import* 
from Models.StateSpace import* 
from Models.RNN import* 
from Operators.Ensemble import* 
from Operators.Boosting import* 
from Operators.CyclicControl import* 
from Evaluation.Evaluate import* 
import pandas as pd
Name="ControllerTrial"
import time

m = Motor()

dT=0.1
m.setTimeStep(dT)
time_step=1
output_time_step=1

input_size=3
output_size=2

model1 = ControlDenseModel(time_step=time_step,output_time_step=output_time_step,input_size=5,output_size=1)
model2 = SSModel(time_step=time_step,output_time_step=output_time_step,input_size=3,output_size=2)
model3 = DenseModel(time_step=time_step,output_time_step=output_time_step,input_size=3,output_size=2)

X=[]
y=[]

moving_input=np.zeros((time_step,input_size))
moving_output=np.zeros((output_time_step,output_size))
controlInput=np.zeros((1,1))
for i in np.arange(0,25000):
    # control input function
	if i>0 and i%100==0:
		print ("Loop Number-",i)
	stateTensor=m.state
	controlInput[:]=m.getControlInput()
	stateTensor=np.append(stateTensor,controlInput,-1)
	outBar=m.step(controlInput[0][0])

	if time_step>1: #To shift Inputs to the left
		moving_input[0:time_step-1,:]=moving_input[1:time_step,:]

	moving_input[time_step-1,:]=stateTensor
	moving_input2=np.asarray(list(moving_input))

	if output_time_step>1:
		moving_output[0:output_time_step-1,:]=moving_output[1:output_time_step,:]

	moving_output[output_time_step-1,:]=outBar-stateTensor[:,0:output_size]
	moving_output2=np.asarray(list(moving_output))

	X.append(moving_input2)
	y.append(moving_output2)

X=np.asarray(X)
y=np.asarray(y)
model2.fit(X,y,epochs=20,batch_size=32)

model2.trainable=False

model,modeltrue=CyclicControlModel(model1,model2,model3,input_size=2)
print (model.summary())

m.reset()
stateTensor=m.state
stateTensor=stateTensor.reshape(1,1,2)
ref=np.array([1,0.2])
ref=ref.reshape(1,1,2)
controlInput=np.zeros((1,1,1))
stateNew=np.array(stateTensor)
X=[]
y=[]
for i in np.arange(0,25000):
	print (ref,stateTensor,stateNew,controlInput,i)
	time.sleep(0.1)

	controlInput,stateTensor,_=modeltrue.predict([stateTensor,ref-stateTensor,controlInput])
	X.append([stateTensor,ref-stateTensor,controlInput])
	y.append([ref[:,:,:],stateTensor])
	model.fit([stateTensor,ref-stateTensor,controlInput],[ref[:,:,:],stateTensor],epochs=100,verbose=1)
	stateNew=np.array(stateTensor)	
	stateTensor=m.step(controlInput[0][0][0])
	stateTensor=stateTensor.reshape(1,1,2)

	if i%50==0:
		for j in range(len(X)):
			model.fit(X[j],y[j],epochs=10,verbose=1)

	if i==500:
		ref[:,:,0]=-100	
	elif i==1000:
		ref[:,:,0]=100	
