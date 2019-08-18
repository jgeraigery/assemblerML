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
Name="10Input1OutputDense-W-512-D-10"

DataFrame=pd.read_csv("SphereparticlesExtrapolation - 2.tsv",sep='\t',header=0)

print(DataFrame.columns)

DFT=DataFrame[DataFrame['mode']=="Tracking"]

DFT.drop(columns=["mode","ctrl.img.filename","id","type","frame"])

DataFrame=DFT

time=np.asarray(DataFrame["t, s"])

particle=np.asarray(DataFrame[["Px","Py","Pt"]])
sprite=np.asarray(DataFrame[["Sx","Sy","St"]])
target=np.asarray(DataFrame[["Tx","Ty","Tt"]])

time_step=1
output_time_step=1

input_size=10
output_size=3

X=[]
y=[]
out=[]
moving_input=np.zeros((time_step,input_size))
moving_output=np.zeros((output_time_step,output_size))
moving_pred=np.zeros((output_time_step,output_size))

for i in np.arange(0,len(time)-1):
    # control input function
	#if np.abs((time[i+1]-time[i])-0.017)>0.005:
	if False:
		continue
	else:
		stateTensor=np.append(particle[i],sprite[i])
		stateTensor=np.append(stateTensor,target[i])
		stateTensor=np.append(stateTensor,time[i+1]-time[i])
		outBar=particle[i+1]
		outBar1=sprite[i]

		if time_step>1: #To shift Inputs to the left
			moving_input[0:time_step-1,:]=moving_input[1:time_step,:]
	
		moving_input[time_step-1,:]=stateTensor
		moving_input2=np.asarray(list(moving_input))
	
		if output_time_step>1:
			moving_output[0:output_time_step-1,:]=moving_output[1:output_time_step,:]
	
		moving_output[output_time_step-1,:]=outBar-stateTensor[0:3]
		moving_output2=np.asarray(list(moving_output))


		if output_time_step>1:
			moving_pred[0:output_time_step-1,:]=moving_pred[1:output_time_step,:]

		angleRad=outBar1[2]*np.pi/180
		alphaRad=(-outBar1[2]+90)*np.pi/180

		if(np.abs(np.cos(alphaRad)) < 0.0001):
			maxDist=1
		else:
			maxDist=np.abs(np.cos(alphaRad))*np.abs(np.tan(alphaRad)*stateTensor[0]-stateTensor[1]-np.tan(alphaRad)*outBar1[0]+outBar1[1])
		
		moving_pred[output_time_step-1,0]=3*(min(maxDist,1)*np.cos(angleRad))
		moving_pred[output_time_step-1,1]=3*(min(maxDist,1)*np.sin(angleRad))
		moving_pred[output_time_step-1,2]=(stateTensor[2]-stateTensor[2])
		moving_pred2=np.asarray(list(moving_pred))
	
		X.append(moving_input2)
		y.append(moving_output2)
		out.append(moving_pred2)

X=np.asarray(X)

print (X.shape)
np.save("1Input1OutputSphereX2.npy",X)

y=np.asarray(y)
out=np.asarray(out)
np.save("1Input1OutputSpherey2.npy",y)
np.save("1Input1OutputSpherephysics2.npy",out)

#print (X.stop)

for i in range(3):
	print("Alpha 0.5 RMSE Error of State:",i,np.sqrt(np.mean((out[:,:,i]-y[:,:,i])**2)))
	print("Alpha 0.5 RMSE Error of State:",i,np.std(np.sqrt((out[:,:,i]-y[:,:,i])**2)))
	#print("Alpha 0.5 R2 of State:",i,np.corrcoef(out[:,:,i].T,y[:,:,i].T))
	#print("Spearmanr of State:",i,spearmanr(out[:,:,i],y[:,:,i]))




outresults=[]
out=np.zeros((1,time_step,input_size))
out[:,:,:]=X[0,:,:]
outresults.append(np.array(out[:,[0],:]))



for i in range(len(X)-1):
	angleRad=out[:,-1,5]*np.pi/180
	alphaRad=(out[:,-1,5]+90)*np.pi/180

	if(np.abs(np.cos(alphaRad)) < 0.0001):
		maxDist=3
	else:
		maxDist=np.abs(np.cos(alphaRad))*np.abs(np.tan(alphaRad)*out[:,-1,0]-out[:,-1,1]-np.tan(alphaRad)*out[:,-1,3]+out[:,-1,4])
		
        out[:,0:time_step-1,:]=np.array(out[:,1:time_step,:])
	out[:,-1,0]+=3*(min(maxDist,1)*np.cos(angleRad))
	out[:,-1,1]+=3*(min(maxDist,1)*np.sin(angleRad))
        out[:,:,2:]=X[i+1,:,2:]
        if i%1==0 and i>0:
                out[:,:,0:2]=X[i+1,:,0:2]+i/100.0
        outresults.append(np.array(out[:,[0],:]))
        #time.sleep(2)
        print("Loop,Predicted,True",i,out[:,-1,0:3],X[i+1,-1,0:3])
        #out[:,:,1]=X[i+1,:,1]
        #out[:,:,0]=X[i+1,:,0]

outresults=np.array(outresults)
outresults=np.reshape(outresults,(len(outresults),1,input_size))


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
ani.save("testphysics.avi", writer=mywriter)

