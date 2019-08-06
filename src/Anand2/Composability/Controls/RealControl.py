# Please change the root variable as required.
# train and test directories of the dataset need to be present in the root directory
# weights need to be present in a weights folder in the root directory

from Problems.NNDelay import *
from numpy import sin,cos,pi
from Models.ControlDense import* 
from Models.Dense import* 
from Models.StateSpace import* 
from Models.RNN import* 
from Operators.Ensemble import* 
from Operators.Boosting import* 
from Operators.RealControl import* 
from Evaluation.EvaluateControl import* 
import pandas as pd


import nidaqmx
import serial
import time
import numpy as np

Name="RealController"
COMPORT = 'COM11'


time_step=1
output_time_step=1

input_size=3
output_size=2

model1 = DenseModel(time_step=time_step,output_time_step=output_time_step,input_size=input_size*2-1,output_size=1,output_function="sigmoid",depth=3,scaling=10)	#Controller

model2 = DenseModel(time_step=time_step,output_time_step=output_time_step,input_size=input_size,output_size=output_size,output_function="linear",depth=3,width=10)	#SystemID Model

model3 = DenseModel(time_step=output_time_step,output_time_step=time_step,input_size=output_size,output_size=input_size,output_function="linear",depth=3)	#Inverse Function


model2.load_weights(""1Input1OutputDenseRealData.hdf5"") #Loading Weights

model2.trainable=False	#Freezing SystemID Model


model,modeltrue=CyclicControlModel(model1,model2,model3,input_size=1,time_step=time_step) #Creating Controller Model




ard = serial.Serial(COMPORT, 52600, timeout=.1)
#time.sleep(1) #give the connection a second to settle
# arduino.write("Hello from Python!")

#Measurement Configuration
encoderTask = nidaqmx.Task()
encoderChan = encoderTask.ci_channels.add_ci_ang_encoder_chan("Dev1/ctr0",pulses_per_rev=300)
encoderChan.ci_encoder_a_input_term="/Dev1/PFI8"
encoderChan.ci_encoder_b_input_term="/Dev1/PFI10"
print(encoderChan.ci_encoder_a_input_term)
print(encoderChan.ci_encoder_b_input_term)

inputTask = nidaqmx.Task()
inputChan = inputTask.ai_channels.add_ai_voltage_chan("Dev1/ai4")
inputTask.timing.samp_clk_rate=20000
#print(inputTask.timing.samp_clk_rate)

inputTask.start()
encoderTask.start()
lastPos = 0
datArray = np.zeros((1,1,3))
state = np.zeros((1,1,3))

currTime = time.clock()
startTime = time.clock()
print(currTime)
true=True

ref=np.array([10.0,0.0])
ref=ref.reshape(1,1,2)
controlInput=np.zeros((1,1,1))
moving_input=np.zeros((1,1,2))


while true:
	print("Reference,CurrentState:"ref,state)
	controlInput[:,-1,:],stateTensor,_=modeltrue.predict([moving_input,ref,controlInput])
	controlInput=np.around(controlInput,decimals=4)
	model.fit([moving_input,ref,controlInput],[ref[:,[-1],:],np.concatenate([moving_input,controlInput],axis=2)],epochs=100,verbose=0)
	arduino.write("Q"+str(controlInput[:,:,:])+"\n")
	raw = inputTask.read(number_of_samples_per_channel=40)
	state[:,:,2] = np.mean(raw)
	state[:,:,0] = encoderTask.read(number_of_samples_per_channel=1)
	state[:,:,1] = (state[:,:,0]-lastPos)/(2.5*10-3)
	lastPos = np.array(state[:,:,0])
	datArray=np.append(datArray,state,0)

	moving_input[:,-1,:]=np.array(state)


	while time.clock() < currTime + .002:
		pass
	currTime = time.clock()
	if currTime-startTime>10:
		true=False
"""
while True:
	try:
		data = arduino.readline()
		try:
			decoded_data= data.rstrip('\n')
			decoded_data= float(decoded_data.decode("utf-8"))
			decoded_data=np.reshape(np.asarray(decoded_data),(1,1,3))
    		pred=model.predict(decoded_data)
			arduino.write(str(pred))
		except:
			print("Error in Decoding Data")
			continue
	except:
		print("Error in Reading Data")
"""

ard.close()



'''
stateTensor=m.state
stateTensor=stateTensor.reshape(1,1,1)
ref=np.array([10.0])
ref=ref.reshape(1,1,1)
ref=np.repeat(ref,time_step,axis=1)
controlInput=np.zeros((1,time_step,1))
stateNew=np.array(stateTensor)
PrevPosition=np.array(stateTensor)

moving_input=np.zeros((1,time_step,input_size-1))
moving_input[:,-1,:]=np.array(stateTensor)

for i in np.arange(0,800):
	#print ("Reference State,True State, Previous State, Predicted State,Control Action,Loop Count",ref,stateTensor,PrevPosition,stateNew,controlInput,i)
	print ("Reference State,True State,Previous State,Predicted State,Control Action,Loop Count",ref[:,0,:],stateTensor[:,:,:],PrevPosition[:,-1,:],stateNew[:,-1,:],controlInput[:,-1,:],i)
	#time.sleep(0.1)
	if i<=200:
		ref[:,-1,:]=3
	elif i>200 and i<=400:
		ref[:,-1,:]=30
	elif i>400 and i<=500:
		ref[:,-1,:]=100
	elif i>500 and i<=600:
		ref[:,-1,:]=10
	else:
		ref[:,-1,:]-=10*dT
		

	#controlInput,stateTensor,PrevPosition=modeltrue.predict([stateTensor,ref-stateTensor,controlInput])
	controlInput[:,-1,:],stateTensor,PrevPosition=modeltrue.predict([moving_input,ref,controlInput])
	#controlInput,stateTensor=modeltrue.predict([stateTensor,ref,controlInput])
	X1.append(np.array(stateTensor))
	X2.append(np.array(ref[:,-1,:]))
	X3.append(np.array(controlInput[:,-1,:]))
	moving_input[:,-1,:]=np.array(stateTensor)



	model.fit([moving_input,ref,controlInput],[ref[:,[-1],:],np.concatenate([moving_input,controlInput],axis=2)],epochs=100,verbose=0)
	stateNew=np.array(stateTensor)	
	stateTensor=m.step(controlInput[:,-1,:])
	stateTensor=stateTensor.reshape(1,1,1)
	moving_input[:,-1,:]=np.array(stateTensor)

	if time_step>1: #To shift Inputs to the left
		moving_input[:,0:time_step-1,:]=np.array(moving_input[:,1:time_step,:])
		controlInput[:,0:time_step-1,:]=np.array(controlInput[:,1:time_step,:])
		ref[:,0:time_step-1,:]=np.array(ref[:,1:time_step,:])

		

X1=np.reshape(np.asarray(X1),(len(X1),1))
X2=np.reshape(np.asarray(X2),(len(X2),1))
X3=np.reshape(np.asarray(X3),(len(X3),1))

X=np.concatenate([X1,X2,X3],axis=-1)

evaluate(X,name="Images/NN-Delay-CyclicControl-M1RNN-M2RNN-M3RNN-LookBack10",output_size=1)
'''
