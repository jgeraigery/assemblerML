# Please change the root variable as required.
# train and test directories of the dataset need to be present in the root directory
# weights need to be present in a weights folder in the root directory

import gym
from gym import spaces


from numpy import sin,cos,pi
from Models.Dense import* 
from Models.StateSpace import* 
from Models.RNN import* 
from Operators.Ensemble import* 
from Operators.Boosting import* 
from Evaluation.Evaluate import* 
import random

np.random.seed(1)
random.seed(1)
Name="Dense"

m = gym.make("Pendulum-v0",g=9.8)
m.seed(1)
#m.dt=2.0
#m.g=2.0

time_step=1
output_time_step=1

input_size=4
output_size=3

simulation_time=50000
training_time=10000
control_frequency=100

train_on_fly=True

X=[]
y=[]

moving_input=np.zeros((time_step,input_size))
moving_output=np.zeros((output_time_step,output_size))
controlInput=np.zeros((1))

for i in range(25000):
	# control input function
        if i>0 and i%20000==0:
                print ("Loop Number-",i)
                print ("ControlInput-",controlInput)
		
        if i%10000000==0:
                stateTensor=m.reset()
                print(m.state)
        else:
                stateTensor=outBar
       	controlInput[:]=(np.random.rand(1)*4)-2
        stateTensor=np.append(stateTensor,controlInput)
        outBar,_,_,_=m.step(controlInput)
        #outBar=m.state
        if time_step>1: #To shift Inputs to the left
                moving_input[0:time_step-1,:]=moving_input[1:time_step,:]

        moving_input[time_step-1,:]=stateTensor
        moving_input2=np.copy((moving_input))

        if output_time_step>1:
                moving_output[0:output_time_step-1,:]=moving_output[1:output_time_step,:]

        moving_output[output_time_step-1,:]=outBar-stateTensor[0:output_size]
        #moving_output[output_time_step-1,:]=outBar
        moving_output2=np.copy((moving_output))

        X.append(moving_input2)
        y.append(moving_output2)

X=np.asarray(X)
y=np.asarray(y)

np.save("2Torque_20HZ_Data.npy",X)
#np.save("1Torque_20HZ_Label.npy",y)
