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

#Writing Parser for Python File
parser = OptionParser()

parser.add_option("-p", "--path", dest="test_path", help="Path to test data in the form of tsv")
parser.add_option("-l", "--look_back", type="int", dest="look_back",help="Number of previous time_steps used when  training model ", default=1)
parser.add_option("-f", "--predict_forward", type="int", dest="predict_forward",help="Number of future predicted time_steps when training model ", default=1)
parser.add_option("--network", dest="network", help="Path to  Trained H5 File", default=None)

(options, args) = parser.parse_args()

if not options.test_path:   # if filename is not given
        parser.error('Error: path to test data must be specified. Pass --path to command line')

if not options.network:   	# if Network is
        parser.error('Error: Please pass test Network')

#Taking Input annd Output Time_Step
time_step=options.look_back
output_time_step=options.predict_forward

#Parsing Input File
Name=options.test_path

DataFrame=pd.read_csv(Name,sep='\t',header=0)

#Using Only Tracking Datapoints
DFT=DataFrame[DataFrame['mode']=="Tracking"]
#Removing Unrequired columns
DFT.drop(columns=["mode","ctrl.img.filename","id","type","frame"])
DataFrame=DFT

#Taking Time, Poisitions of Particle,Sprite,Target as numpy arrays
time=np.asarray(DataFrame["t, s"])
particle=np.asarray(DataFrame[["Px","Py","Pt"]])
sprite=np.asarray(DataFrame[["Sx","Sy","St"]])
target=np.asarray(DataFrame[["Tx","Ty","Tt"]])

#Input[Px,Py,Pt,Sx,Sy,St,Tx,Ty,Tt,DeltaTime] and Output [PxDeltaT,PyDeltaT,PtDeltaT] Size for the Model
input_size=10
output_size=3


#Preprocessing to get input in proper shape for the Deep Model
X=[]
moving_input=np.zeros((1,input_size))

for i in np.arange(1,len(time)-1):
        #if np.abs((time[i+1]-time[i])-0.017)>0.005: #Use if you want to  make frequency of input to be the same
        if False:
                continue
        else:
                stateTensor=np.append(particle[i],sprite[i])
                stateTensor=np.append(stateTensor,target[i])
                stateTensor=np.append(stateTensor,time[i]-time[i-1])
                moving_input[:,:]=stateTensor
                moving_input2=np.asarray(list(moving_input))

                X.append(moving_input2)

X=np.asarray(X)


X2=np.zeros((len(X)-2*time_step,time_step,input_size)) # Restructuring X based on Input Time Step and Y based on Output_time_step Note-Should do this more efficiently
y=np.zeros((len(X)-2*time_step,output_time_step,output_size))
for i in range(time_step,len(X)-time_step):
	X2[i-time_step,:,:]=np.array(np.reshape(X[i-time_step:i],(1,time_step,10)))
        y[i-time_step,:,:]=np.array(np.reshape(X[i:i+output_time_step,:,0:3]-X[i-1:i+output_time_step-1,:,0:3],(1,output_time_step,3)))

X=np.array(X2)

model=load_model(options.network)


#Saving Weight File for Evaluation
Name=Name.replace(".tsv","")
#Remvoing Strings if Previously saved H5 file
Base=options.network.replace("Weights/","")
Base=Base.replace(".h5","")
Base=Base.replace(Name,"")
Name=Name + "-Base-" + Base


out=model.predict(X)

if type(model.output_shape) is list:
	out=out[0]+out[1]

for j in range(1):
	for i in range(3):
		print(" Model-"+Name+" RMSE Error of State:",j,i,np.sqrt(np.mean((out[:,j,i]-y[:,j,i])**2)))
		print("Correlation Coefficinet of State:",i,np.corrcoef(out[:,j,i].T,y[:,j,i].T)[0][1])
		#print("Spearman Rank Correlation of State:",i,spearmanr(out[:,0,i],y[:,0,i]))


result=np.concatenate((out[:,[-1],:],y[:,[-1],:]+X[:,[-1],0:output_size],X[:,[-1],0:output_size]),axis=2) #Concatenating just last time step of Predicted Output, True Output y, and Input X
result=result.reshape(len(X),output_size*3)

time=np.zeros((len(X),1))
for i in range(len(X)):
	time[i]=np.sum(X[0:i+1,-1,-1])

result=np.concatenate((result,time),axis=1)
evaluate(result,output_size=output_size,Training_Time=0,name="Outputs/Testing-"+Name)


