# Please change the root variable as required.
# train and test directories of the dataset need to be present in the root directory
# weights need to be present in a weights folder in the root directory

import gym
from numpy import sin,cos,pi
from Models.Dense import* 
from Models.StateSpace import* 
from Models.RNN import* 
from Operators.Ensemble import* 
from Operators.Boosting import* 
from Evaluation.Evaluate import* 

Name="EnsembleRNN"
m = gym.make("Acrobot-v1")
m.reset()
dT = .001

time_step=1
output_time_step=1

input_size=7
output_size=6

simulation_time=50000
training_time=10000
control_frequency=100

train_on_fly=True

model1 = SSModel(time_step=time_step,output_time_step=output_time_step,input_size=input_size,output_size=output_size)
model2 = SSModel(time_step=time_step,output_time_step=output_time_step,input_size=input_size,output_size=output_size)

model= EnsembleModel(model1,model2,time_step=time_step,input_size=input_size,output_size=output_size)

result = np.zeros([1,output_size*3+2])

X=[]
y=[]

moving_input=np.zeros((time_step,input_size))
moving_output=np.zeros((output_time_step,output_size))

for i in np.arange(0,training_time):
    # control input function
	if (i%control_frequency==0):
		print ("Simulation Time:",i*dT)
		controlInput = m.action_space.sample()

	stateTensor =np.asarray(list(m.state))
	stateTensor=np.asarray([cos(stateTensor[0]), sin(stateTensor[0]), cos(stateTensor[1]), sin(stateTensor[1]), stateTensor[2], stateTensor[3]])
	#stateTensor=np.asarray([cos(s[0]), np.sin(s[0]), cos(s[1]), sin(s[1]), s[2], s[3]])
	stateTensor=np.append(stateTensor,controlInput)
	outBar,_,_,_=m.step(controlInput)
	if time_step>1: #To shift Inputs to the left
		moving_input[0:time_step-1,:]=moving_input[1:time_step,:]

	moving_input[time_step-1,:]=stateTensor
	moving_input2=np.asarray(list(moving_input))

	if output_time_step>1:
		moving_output[0:output_time_step-1,:]=moving_output[1:output_time_step,:]

	moving_output[output_time_step-1,:]=outBar-stateTensor[0:output_size]
	moving_output2=np.asarray(list(moving_output))

	X.append(moving_input2)
	y.append(moving_output2)

X=np.asarray(X)
y=np.asarray(y)
model.fit(X,y,epochs=100,batch_size=32)

m.reset()

X1=[]
y1=[]

moving_input=np.zeros((time_step,input_size))
moving_output=np.zeros((output_time_step,output_size))

for i in np.arange(0,training_time):
    # control input function
	if (i%control_frequency==0):
		print ("Simulation Time:",i*dT)
		controlInput = m.action_space.sample()

	stateTensor =np.array(list(m.state))
	stateTensor=np.asarray([cos(stateTensor[0]), sin(stateTensor[0]), cos(stateTensor[1]), sin(stateTensor[1]), stateTensor[2], stateTensor[3]])
	stateTensor=np.append(stateTensor,controlInput)
	outBar,_,_,_=m.step(controlInput)

	if time_step>1: #To shift Inputs to the left
		moving_input[0:time_step-1,:]=moving_input[1:time_step,:]

	moving_input[time_step-1,:]=stateTensor
	moving_input2=np.asarray(list(moving_input))

	if output_time_step>1:
		moving_output[0:output_time_step-1,:]=moving_output[1:output_time_step,:]


	moving_output[output_time_step-1,:]=outBar-stateTensor[0:output_size]
	moving_output2=np.asarray(list(moving_output))

	X1.append(moving_input2)
	y1.append(moving_output2)

X1=np.asarray(X1)
y1=np.asarray(y1)

out=model.predict(X1)

result=np.concatenate((out,y1+X1[:,:,0:output_size],X1),axis=2)

result=result.reshape(10000,output_size*3+1)
evaluate(result,output_size=output_size,Training_Time=0,name="Images/Interpolation"+Name)

m.reset()

X1=[]
y1=[]

moving_input=np.zeros((time_step,input_size))
moving_output=np.zeros((output_time_step,output_size))

for i in np.arange(0,training_time):
    # control input function
	if (i%10==0):
		print ("Simulation Time:",i*dT)
		controlInput = m.action_space.sample()

	stateTensor=np.array(list(m.state))
	stateTensor=np.asarray([cos(stateTensor[0]), sin(stateTensor[0]), cos(stateTensor[1]), sin(stateTensor[1]), stateTensor[2], stateTensor[3]])
	stateTensor=np.append(stateTensor,controlInput)
	outBar,_,_,_=m.step(controlInput)

	if time_step>1: #To shift Inputs to the left
		moving_input[0:time_step-1,:]=moving_input[1:time_step,:]

	moving_input[time_step-1,:]=stateTensor
	moving_input2=np.asarray(list(moving_input))

	if output_time_step>1:
		moving_output[0:output_time_step-1,:]=moving_output[1:output_time_step,:]


	moving_output[output_time_step-1,:]=outBar-stateTensor[0:output_size]
	moving_output2=np.asarray(list(moving_output))

	out=np.zeros((1,output_time_step,output_size))
	X1.append(moving_input2)
	y1.append(moving_output2)

X1=np.asarray(X1)
y1=np.asarray(y1)

out=model.predict(X1)

result=np.concatenate((out,y1+X1[:,:,0:output_size],X1),axis=2)

result=result.reshape(10000,output_size*3+1)
evaluate(result,output_size=output_size,Training_Time=0,name="Images/Extrapolation"+Name)
