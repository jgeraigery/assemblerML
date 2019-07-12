# Please change the root variable as required.
# train and test directories of the dataset need to be present in the root directory
# weights need to be present in a weights folder in the root directory

import gym,tensorflow as tf
from gym import spaces, logger
from numpy import sin,cos,pi
from Models.Dense import* 
from Models.StateSpace import* 
from Models.RNN import* 
from Operators.Ensemble import* 
from Operators.Boosting import* 
from Evaluation.Evaluate import* 
import time

Name="SSBoosted"
m = gym.make("CartPole-v0")

dT = .001

time_step=1
output_time_step=1

input_size=5
output_size=4

simulation_time=50000
training_time=200
control_frequency=1

train_on_fly=True

def customLoss(input1,input2):
	def loss(y_true,y_pred):
		input=K.square(input1-input2)
		return K.mean(tf.where(tf.is_nan(input),tf.zeros_like(input),input))+K.square(y_true-y_pred)		
	return loss

input1 = Input(batch_shape=(None,4))
input2 = Input(batch_shape=(None,4))
input3 = Input(batch_shape=(None,1))
input=keras.layers.concatenate([input1,input3],-1)
x = Dense(100,activation="relu")(input)
x = Dense(100,activation="relu")(x)
output = Dense(1,activation="sigmoid")(x)
model = Model(inputs=[input1,input2,input3], outputs=[output])

model.compile(loss=customLoss(input1,input2), optimizer="adam", metrics=['accuracy'])


ref=np.array([0.0,np.nan,0.0,np.nan])
label=np.array([0.0])
label=np.zeros((1000,1))
label[0:500]=1
rewards=np.zeros((100))

X=[]
y=[]
y2=[]

controlInput=np.zeros((1,1),dtype='int')
for resets in range(1000):
	outBar=m.reset()
	for i in np.arange(0,training_time):
    	# control input function
		if (i%control_frequency==0):
			print ("Simulation Time:",i*dT)
			controlInput[:] = np.int(np.round(model.predict([outBar.reshape(1,4),np.zeros((1,4)),controlInput.reshape(1,1)])))
			print ((controlInput))
		outBar,reward,done,_=m.step((controlInput[0][0]))
		print (outBar,controlInput,reward)
		X.append(outBar)
		y.append(ref)
		y2.append(controlInput[0][0])
		#model.fit([np.repeat(outBar,1000).reshape(1000,4),np.repeat(reward,1000).reshape(1000,1)],label,epochs=10,verbose=0,batch_size=1000)
		if done:
			#for trail in range(1000):
			X.append(outBar)
			y.append(ref)
			y2.append(controlInput[0][0])
			labelnew=np.zeros((len(y),1))
			labelnew[0:int(len(y)/2)]=1
			model.fit([np.asarray(X),np.asarray(y),np.asarray(y2)],labelnew,epochs=100,verbose=0,batch_size=len(X))
			rewards[resets]=(i+1)
			print ("Number of Iterations Lasted:",i)
			time.sleep(1)
			break	

print ("Total Reward",np.mean(rewards))
#print ("Reward After Training Phase",np.mean(rewards[100:]))
