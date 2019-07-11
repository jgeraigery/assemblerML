# Please change the root variable as required.
# train and test directories of the dataset need to be present in the root directory
# weights need to be present in a weights folder in the root directory

import gym
from numpy import sin,cos,pi
from Models.Dense import* 
from Models.ControlDense import* 
from Models.StateSpace import* 
from Models.RNN import* 
from Operators.Ensemble import* 
from Operators.Boosting import* 
from Operators.Control import* 
from Evaluation.Evaluate import* 

Name="Controls"
m = gym.make("CartPole-v0")
m.reset()
dT = .001

time_step=1
output_time_step=1

input_size=5
output_size=4

simulation_time=50000
training_time=10000
control_frequency=100

train_on_fly=True

model1 = SSModel(time_step=time_step,output_time_step=output_time_step,input_size=input_size,output_size=output_size)
model2 = SSModel(time_step=time_step,output_time_step=output_time_step,input_size=input_size,output_size=output_size)

model3 = ControlDenseModel(time_step=time_step,output_time_step=output_time_step,input_size=input_size,output_size=1)

model= EnsembleModel(model1,model2,time_step=time_step,input_size=input_size,output_size=output_size)

result = np.zeros([1,output_size*3+2])

X=[]
y=[]

moving_input=np.zeros((time_step,input_size))
moving_output=np.zeros((output_time_step,output_size))

for i in np.arange(0,training_time):
    # control input function
	print ("Simulation Time:",i*dT)
	controlInput = m.action_space.sample()
	if (i%control_frequency==0):
		m.reset()
	stateTensor =np.asarray(list(m.state))
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

modelnew=ControlModel(model3,model,time_step=time_step,input_size=input_size,output_size=output_size)

ref=np.array([0.0,np.nan,0.0,np.nan])
currentpos=np.array([0.0,0.0,0.0,0.0,0])
label=np.array([0.0])
label=np.zeros((1000,1))
label[0:500]=1
rewards=np.zeros((150))

for resets in range(1):
	for i in np.arange(0,training_time):
	# control input function
		if (i%control_frequency==0):
			print ("Simulation Time:",i*dT)
		controlinput,currentpos=modelnew.predict([currentpos.reshape(1,1,5)])
	print(controlinput,currentpos)
		
