# Please change the root variable as required.
# train and test directories of the dataset need to be present in the root directory
# weights need to be present in a weights folder in the root directory

from numpy import sin,cos,pi
from Models.ControlDense import* 
from Models.Dense import* 
from Models.StateSpace import* 
from Models.RNN import* 
from Operators.Ensemble import* 
from Operators.Boosting import* 
from Operators.TSCyclicControl import* 
from Evaluation.EvaluateControl import* 
import pandas as pd

import gym

Name="IPController"
import time

m = gym.make("Pendulum-v0")
#stateTensor=m.reset()
control_time_step=10
time_step=1
output_time_step=1

input_size=4
output_size=3

#model1 = ControlDenseModel(time_step=time_step,output_time_step=output_time_step,input_size=4,output_size=1)
model1 = DenseModel(time_step=control_time_step,output_time_step=output_time_step,input_size=7,depth=3,width=50,output_size=1,output_function='sigmoid') #Controller Model
model2 = DenseModel(time_step=output_time_step,output_time_step=output_time_step,input_size=4,depth=3,output_size=3)
model3 = DenseModel(time_step=time_step,output_time_step=output_time_step,input_size=4,depth=3,output_size=3)

X=[]
y=[]

moving_input=np.zeros((output_time_step,input_size))
moving_output=np.zeros((output_time_step,output_size))
controlInput=np.zeros((1))
for i in np.arange(0,25000):
    # control input function
	if i>0 and i%100==0:
		print ("Loop Number-",i)

	if i%1000==0:
		stateTensor=m.reset()		
	else:
		stateTensor=outBar
	controlInput[:]=m.action_space.sample()
	stateTensor=np.append(stateTensor,controlInput)
	outBar,_,_,_=m.step(controlInput)
	#outBar=m.state
	if time_step>1: #To shift Inputs to the left
		moving_input[0:time_step-1,:]=moving_input[1:time_step,:]

	moving_input[time_step-1,:]=stateTensor
	moving_input2=np.asarray(list(moving_input))

	if output_time_step>1:
		moving_output[0:output_time_step-1,:]=moving_output[1:output_time_step,:]

	#moving_output[output_time_step-1,:]=outBar-stateTensor[:,0:output_size]
	moving_output[output_time_step-1,:]=outBar
	moving_output2=np.asarray(list(moving_output))

	X.append(moving_input2)
	y.append(moving_output2)

X=np.asarray(X)
y=np.asarray(y)

model2.fit(X,y,epochs=40,batch_size=32)

model2.trainable=False

print (model2.summary())
print (model2.get_weights())


model,modeltrue=TSCyclicControlModel(model1,model2,model3,time_step=control_time_step,input_size=3)
print (model.summary())

ref=np.array([np.cos(0),np.sin(0),0.0])
ref=ref.reshape(1,1,3)
ref=np.repeat(ref,control_time_step,axis=1)
controlInput=np.zeros((1,control_time_step,1))
controlInput[:]=m.action_space.sample()
reward=0
rewardsum=[]
for i in np.arange(0,1000):
	if i%100==0:
		print ("State Reset")
		stateTensor=m.reset()
		#m.state=np.array([0.0,0.0])
		#stateTensor=np.array([np.cos(m.state[0]), np.sin(m.state[0]), m.state[1]])
		#controlInput[:]=0
		stateTensor=stateTensor.reshape(1,1,3)
		moving_state_input=np.zeros((1,control_time_step,3))
		stateNew=np.array(stateTensor)
		PrevPosition=np.array(stateTensor)
	'''
	if i>0 and i%100==0:
		if np.sum(rewardsum[(i-100):(i)])<400:
			model1.trainable=False
	'''

	#print ("Reference State,True State, Previous State, Predicted State,Control Action,Loop Count",ref,stateTensor,PrevPosition,stateNew,controlInput,i)
	print ("Reference State,True State,Predicted State,Control Action,Reward,Loop Count",ref[:,0,:],stateTensor,stateNew,controlInput[:,-1,:],reward,i)
	time.sleep(0.1)
		
		
	#controlInput,stateTensor,PrevPosition=modeltrue.predict([stateTensor,ref-stateTensor,controlInput])
	#controlInput,stateTensor,PrevPosition=modeltrue.predict([stateTensor,ref,controlInput])

	controlInput[:,-1,:],stateTensor=modeltrue.predict([moving_state_input,ref,controlInput])

	if control_time_step>1: #To shift Inputs to the left
		moving_state_input[:,0:control_time_step-1,:]=moving_state_input[:,1:control_time_step,:]
		controlInput[:,0:control_time_step-1,:]=controlInput[:,1:control_time_step,:]

	moving_state_input[:,control_time_step-1,:]=np.array(stateTensor)
	model.fit([moving_state_input,ref,controlInput],[ref[:,[0],:]],epochs=100,verbose=0)



	stateNew=np.array(stateTensor)	
	stateTensor,reward,_,_=m.step(controlInput[0][0])
	rewardsum.append(reward)
	stateTensor=stateTensor.reshape(1,1,3)

	moving_state_input[:,control_time_step-1,:]=np.array(stateTensor)
	#controlInput[:,control_time_step-1,:]=np.array(controlInput[0][0][0])

		
print ("Total Reward over 800 Iteration Mean",np.sum(rewardsum))

for i in range(8):
	print (np.sum(rewardsum[i*100:(i+1)*100]))

