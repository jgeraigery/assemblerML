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
Name="10Input10Output-Sphere-RNNBoostedSS-W-30-D-10"

time_step=10
output_time_step=10

input_size=10
output_size=3


#model1 = RNNModel(time_step=time_step,output_time_step=output_time_step,input_size=input_size,output_size=output_size,lr=0.001,depth=10,width=100)
model1 = SSModel(time_step=time_step,output_time_step=output_time_step,input_size=input_size,output_size=output_size,lr=0.001)
model2 = RNNModel(time_step=output_time_step,output_time_step=output_time_step,input_size=output_size,output_size=output_size,lr=0.001,width=30,depth=10)



X=np.load("1Input1OutputX2.npy")
#X1=np.load("1Input1OutputX3.npy")
#y=np.load("1Input1Outputy3.npy")




X2=np.zeros((len(X)-20,10,10))
y=np.zeros((len(X)-20,10,3))
for i in range(10,len(X)-10):
	X2[i-10,:,:]=np.array(np.reshape(X[i-10:i],(1,10,10)))
        y[i-10,:,:]=np.array(np.reshape(X[i:i+10,:,0:3]-X[i-1:i+9,:,0:3],(1,10,3)))


X=np.array(X2)

'''
X2=np.zeros((len(X1)-20,10,10))
y1=np.zeros((len(X1)-20,10,3))
for i in range(10,len(X1)-10):
	X2[i-10,:,:]=np.array(np.reshape(X1[i-10:i],(1,10,10)))
        y1[i-10,:,:]=np.array(np.reshape(X1[i:i+10,:,0:3]-X1[i-1:i+9,:,0:3],(1,10,3)))


X1=np.array(X2)

X=np.concatenate((X,X1),axis=0)
y=np.concatenate((y,y1),axis=0)
'''

reduce_lr = ReduceLROnPlateau(monitor='loss', factor=0.5,
                              patience=5, min_lr=0.00001,verbose=1)

early_stop=EarlyStopping(monitor='loss', min_delta=0.00005, patience=10, verbose=1)
#model1.fit(X,y,epochs=500,batch_size=512,callbacks=[reduce_lr,early_stop])

pred=model1.predict(X)

model1.trainable=False

model = BoostingModel(model1,model2,time_step=time_step,input_size=input_size,output_size=output_size,lr=0.001)

print (model.summary())


#model.fit(X,[y,y-pred],epochs=500,batch_size=512,callbacks=[reduce_lr,early_stop])


#model.save_weights(Name+".hdf5")
model.load_weights(Name+".hdf5")

Name="10Input10Output-Chiplet-RNNBoostedSS-W-30-D-10"


#out1,out2=model.predict(X)

#out=out1+out2

out=model1.predict(X)
for j in range(1):
	for i in range(3):
		print(" Deep Model "+Name+" RMSE Error of State:",j,i,np.sqrt(np.mean((out[:,j,i]-y[:,j,i])**2)))
		print("Alpha 0.5 R2 of State:",i,np.corrcoef(out[:,j,i].T,y[:,j,i].T))
		print("Spearmanr of State:",i,spearmanr(out[:,0,i],y[:,0,i]))


result=np.concatenate((out[:,[-1],:],y[:,[-1],:]+X[:,[-1],0:output_size],X[:,[-1],0:output_size]),axis=2)

result=result.reshape(len(X),output_size*3)

#time=time.reshape(len(time),1)
#time=time[:len(time)-1]

time=np.zeros((len(X),1))

result=np.concatenate((result,time),axis=1)
evaluate(result,output_size=output_size,Training_Time=0,name="Images/TestData"+Name)


#print(stop)

outresults=[]
out=np.zeros((1,10,10))
out[:,:,:]=X[0,:,:]
outresults.append(np.array(out[:,[0],:]))

for i in range(len(X)-1):
        outnew=model1.predict(out)
        #outnew1,outnew2=model.predict(out)
        #outnew=outnew1+outnew2
        out[:,0:9,:]=np.array(out[:,1:10,:])
        out[:,-1,0]+=1*np.array(outnew[:,0,0])
        out[:,-1,1]+=1*np.array(outnew[:,0,1])
        out[:,-1,2]+=1*np.array(outnew[:,0,2])
        out[:,:,3:]=X[i+1,:,3:]
	if i%1000000==0 and i>0:
	        out[:,:,:]=X[i+1,:,:]
        outresults.append(np.array(out[:,[0],:]))
        #time.sleep(2)
        print("Loop,Predicted,True",i,out[:,-1,0:3],X[i+1,-1,0:3])
        #out[:,:,1]=X[i+1,:,1]
        #out[:,:,0]=X[i+1,:,0]

outresults=np.array(outresults)
outresults=np.reshape(outresults,(len(outresults),1,input_size))


for i in range(2):
	print(" Deep Model "+Name+" RMSE Error of State:",i,np.sqrt(np.mean((outresults[:,0,i]-X[:,0,i])**2)))

for i in range(2):
	print(" Deep Model "+Name+" Last 5000 RMSE Error of State:",i,np.sqrt(np.mean((outresults[len(X)-5000:len(X),0,i]-X[len(X)-5000:len(X),0,i])**2)))


plt.figure(1,figsize=(20,10))
plt.subplot(2, 1,1)
plt.plot(outresults[:,0,0],outresults[:,0,1],c="k",linewidth="4",label="Pred")
plt.plot(X[:,0,0],X[:,0,1],c="r",label="True")
plt.ylabel("PY")
plt.xlabel("PX")
plt.legend(loc="upper right")

plt.subplot(2, 1, 2)
plt.plot(X[:,0,3],X[:,0,4],c="r",label="Sprites")
plt.scatter(X[0::1000,0,6],X[0::1000,0,7],c="b",label="Target")
plt.ylabel("Sy")
plt.xlabel("SX")
plt.legend(loc="upper right")
plt.show()
plt.savefig(Name+".svg")
plt.clf()


plt.figure(1,figsize=(20,10))
plt.subplot(2, 1,1)
plt.plot(outresults[0:5000,0,0],outresults[0:5000,0,1],c="k",linewidth="4",label="Pred")
plt.plot(X[0:5000,0,0],X[0:5000,0,1],c="r",label="True")
plt.ylabel("PY")
plt.xlabel("PX")
plt.legend(loc="upper right")

plt.subplot(2, 1, 2)
plt.plot(X[0:5000,0,3],X[0:5000,0,4],c="r",label="Sprites")
plt.scatter(X[0:5000:100,0,6],X[0:5000:100,0,7],c="b",label="Target")
plt.ylabel("Sy")
plt.xlabel("SX")
plt.legend(loc="upper right")
plt.show()
plt.savefig(Name+"ExplodedView.svg")
plt.clf()

