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

from matplotlib.animation import FuncAnimation,FFMpegFileWriter


matplotlib.rcParams['animation.ffmpeg_args'] = '-report'
matplotlib.rcParams['animation.bitrate'] = 2000


import pandas as pd
Name="1Input1Output-Sphere-RNNBoostedPhysics"

time_step=1
output_time_step=1

input_size=10
output_size=3


model = RNNModel(time_step=time_step,output_time_step=output_time_step,input_size=input_size,output_size=output_size,lr=0.001)



X=np.load("1Input1OutputSphereX2.npy")
#X1=np.load("1Input1OutputX3.npy")
y=np.load("1Input1OutputSpherey2.npy")

pred=np.load("1Input1OutputSpherephysics2.npy")




reduce_lr = ReduceLROnPlateau(monitor='loss', factor=0.5,
                              patience=5, min_lr=0.00001,verbose=1)

early_stop=EarlyStopping(monitor='loss', min_delta=0.000005, patience=10, verbose=1)
#model.fit(X,y-pred,epochs=1000,batch_size=512,callbacks=[reduce_lr,early_stop])


print (model.summary())



#model.save_weights(Name+".hdf5")
model.load_weights(Name+".hdf5")



out1=model.predict(X)


out=pred+out1

for j in range(1):
	for i in range(3):
		print(" Deep Model "+Name+" RMSE Error of State:",j,i,np.sqrt(np.mean((out[:,j,i]-y[:,j,i])**2)))
		print(" Deep Model "+Name+" RMSE Error of State:",j,i,np.std(np.sqrt((out[:,j,i]-y[:,j,i])**2)))
		#print("Alpha 0.5 R2 of State:",i,np.corrcoef(out[:,j,i].T,y[:,j,i].T))
		#print("Spearmanr of State:",i,spearmanr(out[:,0,i],y[:,0,i]))


result=np.concatenate((out[:,[-1],:],y[:,[-1],:]+X[:,[-1],0:output_size],X[:,[-1],0:output_size]),axis=2)

result=result.reshape(len(X),output_size*3)

#time=time.reshape(len(time),1)
#time=time[:len(time)-1]

time=np.zeros((len(X),1))

result=np.concatenate((result,time),axis=1)
evaluate(result,output_size=output_size,Training_Time=0,name="Images/TestData"+Name)


print(stop)

outresults=[]
out=np.zeros((1,time_step,input_size))
out[:,:,:]=X[0,:,:]
outresults.append(np.array(out[:,[0],:]))



for i in range(len(X)-1):
        #outnew=model1.predict(out)
        outnew1,outnew2=model.predict(out)
        outnew=outnew1+outnew2
        out[:,0:time_step-1,:]=np.array(out[:,1:time_step,:])
        out[:,-1,0]+=1*np.array(outnew[:,0,0])
        out[:,-1,1]+=1*np.array(outnew[:,0,1])
        out[:,-1,2]+=1*np.array(outnew[:,0,2])
        out[:,:,3:]=X[i+1,:,3:]
	if i%10000000==0 and i>0:
	        out[:,:,:]=X[i+1,:,:]
        outresults.append(np.array(out[:,[0],:]))
        #time.sleep(2)
        #print("Loop,Predicted,True",i,out[:,-1,0:3],X[i+1,-1,0:3])
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

fig, ax = plt.subplots()
xdata, ydata = [], []
xpdata, ypdata = [], []
ln, = plt.plot([], [], 'r', animated=True,label="True")
ln2, = plt.plot([], [], 'k', animated=True,linewidth="4",label="Predicted")

def init():
	ax.set_xlim(0, 1600)
	ax.set_ylim(0, 1600)
	ax.set_ylabel("PY")
	ax.set_xlabel("PX")
	ax.legend()
	ln.set_data(xdata,ydata)
	ln2.set_data(xpdata,ypdata)
	return ln,ln2,

def update(frame):
	xdata.append(frame[0])
	ydata.append(frame[1])
	xpdata.append(frame[2])
	ypdata.append(frame[3])
	ln.set_data(xdata, ydata)
	ln2.set_data(xpdata,ypdata)
	return ln,ln2,

ani = FuncAnimation(fig, update, frames=np.concatenate((X[:10000:10,0,0:2],outresults[:10000:10,0,0:2]),axis=1),init_func=init, blit=True, interval =1,repeat=False)
plt.show()

mywriter = FFMpegFileWriter(fps=25,codec='mpeg4')
ani.save("test.avi", writer=mywriter)
