# Please change the root variable as required.
# train and test directories of the dataset need to be present in the root directory
# weights need to be present in a weights folder in the root directory

from numpy import sin,cos,pi
from Models.Dense import* 
from Models.Arima import* 
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
from keras.models import load_model
from optparse import OptionParser

from matplotlib.animation import FuncAnimation,FFMpegFileWriter
#matplotlib.rcParams['animation.ffmpeg_args'] = '-report'
matplotlib.rcParams['animation.bitrate'] = 2000





#Writing Parser for Python File
parser = OptionParser()

parser.add_option("-p", "--path", dest="test_path", help="Path to test data in the form of npy")
parser.add_option("-l", "--look_back", type="int", dest="look_back",help="Number of previous time_steps used when  training model ", default=1)
parser.add_option("-f", "--predict_forward", type="int", dest="predict_forward",help="Number of future predicted time_steps when training model ", default=1)
parser.add_option("--network", dest="network", help="Path to  Trained Boosting H5 File", default=None)
parser.add_option("--video", dest="video",help="Saves Simulation as Video", default=None)

(options, args) = parser.parse_args()

if not options.test_path:   # if filename is not given
        parser.error('Error: path to test data must be specified. Pass --path to command line')

#Taking Input annd Output Time_Step
time_step=options.look_back
output_time_step=options.predict_forward

#Parsing Input File
Name=options.test_path

input_size=4
output_size=3

X=np.load(Name)
Xp=np.load(Name.replace(".npy","")+"_Physics"+".npy")

X2=np.zeros((len(X)-2*time_step,time_step,input_size)) # Restructuring X based on Input Time Step and Y based on Output_time_step Note-Should do this more efficiently
y=np.zeros((len(X)-2*time_step,output_time_step,output_size))
pred=np.zeros((len(X)-2*time_step,output_time_step,output_size))
for i in range(time_step,len(X)-time_step):
	X2[i-time_step,:,:]=np.array(np.reshape(X[i-time_step:i],(1,time_step,input_size)))
	#y[i-time_step,:,:]=np.array(np.reshape(X[i:i+output_time_step,:,0:3]-X[i-1:i+output_time_step-1,:,0:3],(1,output_time_step,output_size)))
	#pred[i-time_step,:,:]=np.array(np.reshape(Xp[i:i+output_time_step,:,0:3]-Xp[i-1:i+output_time_step-1,:,0:3],(1,output_time_step,output_size)))
	y[i-time_step,:,:]=np.array(np.reshape(X[i:i+output_time_step,:,0:3],(1,output_time_step,output_size)))
	pred[i-time_step,:,:]=np.array(np.reshape(Xp[i:i+output_time_step,:,0:3],(1,output_time_step,output_size)))

X=np.array(X2)

if options.network is not None:
	model=load_model(options.network)
	out=model.predict(X)
	if "Boost" in options.network: 
		out=pred+out
	if "Ensemble" in options.network: 
		out=(0.2*pred+0.8*out)
else:
	out=pred

#Saving Weight File for Evaluation
Name=Name.replace(".npy","")
#Remvoing Strings if Previously saved H5 file
Name=Name+"Physics"
if options.network is not None:
	Base=options.network.replace("Weights/","")
	Base=Base.replace(".h5","")
	Base=Base.replace(Name,"")
	Name=Name + "-Base-" + Base



for j in range(1):
	for i in range(3):
		print(" Model-"+Name+" RMSE Error of State:",j,i,np.sqrt(np.mean((out[:,j,i]-y[:,j,i])**2)))
		print("Correlation Coefficinet of State:",i,np.corrcoef(out[:,j,i].T,y[:,j,i].T)[0][1])
		#print("Spearman Rank Correlation of State:",i,spearmanr(out[:,0,i],y[:,0,i]))


result=np.concatenate((out[:,[-1],:],y[:,[-1],:]+X[:,[-1],0:output_size],X[:,[-1],:]),axis=2) #Concatenating just last time step of Predicted Output, True Output y, and Input X
result=result.reshape(len(X),output_size*3+1)

time=np.arange(len(X))*0.05
time=np.reshape(time,(len(time),1))
result=np.concatenate((result,time),axis=1)
evaluate(result,output_size=output_size,Training_Time=0,name="Outputs/Testing-"+Name)



if options.video is not None:
	print ("Inside Video")
	outresults=[]
	out=np.zeros((1,time_step,input_size))
	out[:,:,:]=X[0,:,:]
	outresults.append(np.array(out[:,[0],:]))
		
	for i in range(len(X)-1):
		outnew=model.predict(out)
		if type(model.output_shape)==list:
			outnew=outnew[0]+outnew[1]
		out[:,0:time_step-1,:]=np.array(out[:,1:time_step,:])
		out[:,-1,0]+=1*np.array(outnew[:,0,0])
		out[:,-1,1]+=1*np.array(outnew[:,0,1])
		out[:,-1,2]+=1*np.array(outnew[:,0,2])
		out[:,:,3:]=X[i+1,:,3:]
		if i%10000000==0 and i>0: #Occasional Datapoint to help tracking
			out[:,:,:]=X[i+1,:,:]
		outresults.append(np.array(out[:,[0],:]))

	outresults=np.array(outresults)
	outresults=np.reshape(outresults,(len(outresults),1,input_size))


	for i in range(2):
        	print(" Deep Model "+Name+" RMSE Error of State:",i,np.sqrt(np.mean((outresults[:,0,i]-X[:,0,i])**2)))

	#Creating Animation of Tracking

	fig, ax = plt.subplots()
	xdata, ydata = [], []
	xpdata, ypdata = [], []
	ln, = plt.plot([], [], 'r', animated=True,label="True")
	ln2, = plt.plot([], [], 'k', animated=True,linewidth="4",label="Predicted")
	#Simple Functions to Help with Plotting
	
	def init():
	        ax.set_xlim(0, 1600)	#Change 0 and 1600 based on expected limits of model
	        ax.set_ylim(0, 1600)	#Change 0 and 1600 based on expected limits of model
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
	ani.save("Outputs/"+Name+".avi", writer=mywriter)


