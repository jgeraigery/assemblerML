# Please change the root variable as required.
# train and test directories of the dataset need to be present in the root directory
# weights need to be present in a weights folder in the root directory

from Problems.DCMotor import *
from Models.Dense import* 
from Models.StateSpace import* 
from Models.RNN import* 
#from Models.LSTM import* 
from Operators.Ensemble import* 
from Operators.Boosting import* 
from Evaluation.Evaluate import* 
import pandas as pd

Name="1Input1OutputSSBoostingDense"


DataFrame=pd.read_excel("RealData/10Hz-2.xlsx")
#DataFrame=pd.read_excel("RealData/1Hz.xlsx")

timeforce=np.asarray(DataFrame["Time"])
force=np.asarray(DataFrame["Force"])

timemotor=np.asarray(DataFrame["Time.1"])
posmotor=np.asarray(DataFrame["Position"])
velomotor=np.asarray(DataFrame["Velocity"])

timemotor=timemotor[~np.isnan(timemotor)]
posmotor=posmotor[~np.isnan(posmotor)]
velomotor=velomotor[~np.isnan(velomotor)]

np.save("timemotor.npy",timemotor)


forceavg=np.zeros(timemotor.shape)
print (timemotor.shape)
print (force)


for i in range(len(forceavg)):
	forceavg[i]=force[i*40:(i+1)*40].mean()
forceavg=np.nan_to_num(forceavg)
print (forceavg,forceavg.shape)

m = Motor()

time_step=10
output_time_step=10

input_size=m.input_size
output_size=m.output_size


X=[]
y=[]

moving_input=np.zeros((time_step,input_size))
moving_output=np.zeros((output_time_step,output_size))

for i in np.arange(0,len(timemotor)-1):
    # control input function
	if i>0 and i%100==0:
		print ("Loop Number-",i)
		print ("CurrentState,NextState",stateTensor,outBar,stateTensor.shape,outBar.shape)
	stateTensor=np.append(posmotor[i],velomotor[i])
	stateTensor=np.append(stateTensor,forceavg[i])
	outBar=np.append(posmotor[i+1],velomotor[i+1])

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



np.save("input10X10Hz.npy",X)
np.save("output10y1Hz.npy",y)
print (stop)


m = Motor()

time_step=1
output_time_step=1

input_size=m.input_size
output_size=m.output_size


model1 = SSModel(time_step=time_step,output_time_step=output_time_step,input_size=input_size,output_size=output_size,lr=0.001)
model2 = DenseModel(time_step=output_time_step,output_time_step=output_time_step,input_size=input_size-1,output_size=output_size,depth=3)
#model3 = RNNModel(time_step=output_time_step,output_time_step=output_time_step,input_size=input_size-1,output_size=output_size)


timemotor=np.load("timemotor.npy")

X=np.load("RealData/X10Hz.npy")
y=np.load("RealData/y10Hz.npy")

X100=np.load("RealData/X100Hz.npy")
y100=np.load("RealData/y100Hz.npy")

X1=np.load("RealData/X1Hz.npy")
y1=np.load("RealData/y1Hz.npy")


model1.fit(X,y,epochs=100,batch_size=32)

pred=model1.predict(X)

model1.trainable=False
model= BoostingModel(model1,model2,time_step=time_step,input_size=input_size,output_size=output_size,lr=0.001)

model.fit(X,[y,y-pred],epochs=100,batch_size=32)

'''
for epochi in range(10):
	for batch in range(780):
		pred=model1.predict(X[batch*32:(batch+1)*32])
		model.fit(X[batch*32:(batch+1)*32],[y[batch*32:(batch+1)*32],y[batch*32:(batch+1)*32]-pred],epochs=1,batch_size=32)
'''



out1,out2=model.predict(X)
#out=model1.predict(X)
#out2=model2.predict(X)
out=out1+out2

result=np.concatenate((out[:,[-1],:],y[:,[-1],:]+X[:,[-1],0:output_size],X[:,[-1],:]),axis=2)

result=result.reshape(len(X),output_size*3+1)

timemotor=timemotor.reshape(len(timemotor),1)
timemotor=timemotor[:len(timemotor)-1]

result=np.concatenate((result,timemotor),axis=1)
evaluate(result,output_size=output_size,Training_Time=0,name="RealData/Images/10Hz"+Name)

#ExtrapolationDataset100HZ

out1,out2=model.predict(X100)
#out=model1.predict(X100)
#out2=model2.predict(X)
out=out1+out2

result=np.concatenate((out[:,[-1],:],y100[:,[-1],:]+X100[:,[-1],0:output_size],X100[:,[-1],:]),axis=2)

result=result.reshape(len(X100),output_size*3+1)

#timemotor=timemotor.reshape(len(timemotor),1)
#timemotor=timemotor[:len(timemotor)-1]

result=np.concatenate((result,timemotor),axis=1)
evaluate(result,output_size=output_size,Training_Time=0,name="RealData/Images/Extrapolation100Hz"+Name)

#ExtrapolationDataset1HZ

out1,out2=model.predict(X1)
#out=model1.predict(X100)
#out2=model2.predict(X)
out=out1+out2

result=np.concatenate((out[:,[-1],:],y1[:,[-1],:]+X1[:,[-1],0:output_size],X1[:,[-1],:]),axis=2)

result=result.reshape(len(X1),output_size*3+1)

#timemotor=timemotor.reshape(len(timemotor),1)
#timemotor=timemotor[:len(timemotor)-1]

result=np.concatenate((result,timemotor),axis=1)
evaluate(result,output_size=output_size,Training_Time=0,name="RealData/Images/Extrapolation1Hz"+Name)

