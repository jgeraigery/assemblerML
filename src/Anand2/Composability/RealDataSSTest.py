# Please change the root variable as required.
# train and test directories of the dataset need to be present in the root directory
# weights need to be present in a weights folder in the root directory

from Problems.DCMotor import *
from Models.Dense import* 
from Models.NALU import* 
from Models.StateSpace import* 
from Models.RNN import* 
from Models.RNNNALU import* 
from Models.NALU import* 
#from Models.LSTM import* 
from Operators.Ensemble import* 
from Operators.Cyclic import* 
from Operators.Boosting import* 
from Evaluation.Evaluate import* 
import pandas as pd
import time
Name="10Input10OutputRNNNTest"

m = Motor()

time_step=10
output_time_step=10

input_size=m.input_size
output_size=m.output_size


model = RNNModel(time_step=time_step,output_time_step=output_time_step,input_size=input_size,output_size=output_size,lr=0.001,width=50,depth=10)

timemotor=np.load("timemotor.npy")

X=np.load("RealData/X1Hz.npy")
#y=np.load("RealData/output10y1Hz.npy")


X2=np.zeros((len(X)-20,10,3))
y=np.zeros((len(X)-20,10,2))
for i in range(10,len(X)-10):
	X2[i-10,:,:]=np.array(np.reshape(X[i-10:i],(1,10,3)))			
	y[i-10,:,:]=np.array(np.reshape(X[i:i+10,:,0:2]-X[i-1:i+9,:,0:2],(1,10,2)))			


X=np.array(X2)

reduce_lr = ReduceLROnPlateau(monitor='loss', factor=0.5,
                              patience=8, min_lr=0.00001)

early_stop=EarlyStopping(monitor='loss', min_delta=0, patience=20, verbose=0)


#model.fit(X,y,epochs=500,batch_size=256,callbacks=[reduce_lr,early_stop])


#print(model1.get_weights())



model.load_weights("10Input10OutputRNNND-10W-50RealData.hdf5")
#model.save_weights("10Input10OutputRNNND-10W-50RealData.hdf5")

outresults=[]
out=np.zeros((1,10,3))
out[:,:,:]=X[0,:,:]
#outresults.append(np.array(out[:,[0],:]))

for i in range(len(X)-1):
	outnew=model.predict(out)
	out[:,0:9,:]=np.array(out[:,1:10,:])
	out[:,-1,0]+=1*np.array(outnew[:,0,0])
	out[:,-1,1]+=1*np.array(outnew[:,0,1])
	out[:,:,2]=X[i+1,:,2]
	outresults.append(np.array(out[:,[0],:]))
	#time.sleep(2)
	print("Predicted,True",out[:,-1,:],X[i+1,-1,:])
	#out[:,:,1]=X[i+1,:,1]
	#out[:,:,0]=X[i+1,:,0]

outresults=np.array(outresults)
outresults=np.reshape(outresults,(len(outresults),1,3))

#outresults[:,:,0]=X[0:len(X)-1,0,[0]]
#outresults[:,:,0]*=50

plt.figure(1,figsize=(20,10))
for i in range(2):
	plt.subplot(3, 1, i+1)
	plt.plot(outresults[:,0,i],c="k",linewidth="4",label="Pred")
	plt.plot(X[:,0,i],c="r",label="True")
	plt.ylabel("State:"+str(i))
	plt.legend(loc="upper right")

plt.subplot(3, 1, 3)
plt.plot(outresults[:,0,2],c="r",label="ControlInput")
plt.ylabel("Control Input")
plt.xlabel("Time")
plt.legend(loc="upper right")
plt.show()
plt.savefig(Name+".svg")
plt.clf()

plt.figure(1,figsize=(20,10))
for i in range(2):
	plt.subplot(3, 1, i+1)
	plt.plot(outresults[0:200,0,i],c="k",linewidth="4",label="Pred")
	plt.plot(X[0:200,0,i],c="r",label="True")
	plt.ylabel("State:"+str(i))
	plt.legend(loc="upper right")

plt.subplot(3, 1, 3)
plt.plot(outresults[0:200,0,2],c="r",label="ControlInput")
plt.ylabel("Control Input")
plt.xlabel("Time")
plt.legend(loc="upper right")
plt.show()
plt.savefig(Name+"ExplodedView.svg")
plt.clf()

