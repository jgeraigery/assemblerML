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

parser.add_option("-p", "--path", dest="training_path", help="Path to training data in the form of tsv")
parser.add_option("-l", "--look_back", type="int", dest="look_back",help="Number of previous time_steps to predict next step ", default=1)
parser.add_option("-f", "--predict_forward", type="int", dest="predict_forward",help="Number of future time_steps to predict", default=1)
parser.add_option("--basenetwork", dest="base_network", help="Base network to use. Supports Physics, Linear, Arima, RNN, Dense or Path to Previous Trained H5 File", default='Linear')
parser.add_option("--boostnetwork", dest="boost_network", help="Boosting network to use. Supports None,Linear,RNN, Dense", default=None)
parser.add_option("--ensemblenetwork", dest="ensemble_network", help="Boosting network to use. Supports None,Linear,RNN, Dense", default=None)

(options, args) = parser.parse_args()

if not options.training_path:   # if filename is not given
        parser.error('Error: path to training data must be specified. Pass --path to command line')

if options.boost_network is not None and options.ensemble_network is not None:   # if filename is not given
        parser.error('Error: Cannot Boost and Ensemble models Simultaneously, use only 1 flag')

#Taking Input annd Output Time_Step
time_step=options.look_back
output_time_step=options.predict_forward

#Parsing Input File
Name=options.training_path

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


#Setting BaseNetwork
if options.ensemble_network is not None or options.boost_network is not None:
	if options.base_network=="Linear":
		model1 = SSModel(time_step=time_step,output_time_step=output_time_step,input_size=input_size,output_size=output_size,lr=0.001)
	elif options.base_network=="RNN":
		model1 = RNNModel(time_step=time_step,output_time_step=output_time_step,input_size=input_size,output_size=output_size,lr=0.001)
	elif options.base_network=="Dense":
		model1 = DenseModel(time_step=time_step,output_time_step=output_time_step,input_size=input_size,output_size=output_size,lr=0.001)
	elif options.base_network=="Arima":
		model1 = ArimaModel(time_step=time_step,output_time_step=output_time_step,input_size=input_size,output_size=output_size,lr=0.001)
	else:
		model1 = load_model(options.base_network)
		model1.name="pretrainedmodel"
else:
	if options.base_network=="Linear":
		model = SSModel(time_step=time_step,output_time_step=output_time_step,input_size=input_size,output_size=output_size,lr=0.001)
	elif options.base_network=="RNN":
		model = RNNModel(time_step=time_step,output_time_step=output_time_step,input_size=input_size,output_size=output_size,lr=0.001)
	elif options.base_network=="Dense":
		model = DenseModel(time_step=time_step,output_time_step=output_time_step,input_size=input_size,output_size=output_size,lr=0.001)
	elif options.base_network=="Arima":
		model = ArimaModel(time_step=time_step,output_time_step=output_time_step,input_size=input_size,output_size=output_size,lr=0.001)
	else:
		model = load_model(options.base_network)
		model.name="pretrainedmodel"


#Setting Second_Network if Ensemble
if options.ensemble_network=="Linear":
	model2 = SSModel(time_step=time_step,output_time_step=output_time_step,input_size=input_size,output_size=output_size,lr=0.001)
elif options.ensemble_network=="RNN":
	model2 = RNNModel(time_step=time_step,output_time_step=output_time_step,input_size=input_size,output_size=output_size,lr=0.001)
elif options.ensemble_network=="Dense":
	model2 = DenseModel(time_step=time_step,output_time_step=output_time_step,input_size=input_size,output_size=output_size,lr=0.001)
elif options.ensemble_network=="Arima":
	model2 = ArimaModel(time_step=time_step,output_time_step=output_time_step,input_size=input_size,output_size=output_size,lr=0.001)

#Setting Second_Network if Boosting
if options.boost_network=="Linear":
	model2 = SSModel(time_step=output_time_step,output_time_step=output_time_step,input_size=output_size,output_size=output_size,lr=0.001)
elif options.boost_network=="RNN":
	model2 = RNNModel(time_step=output_time_step,output_time_step=output_time_step,input_size=output_size,output_size=output_size,lr=0.001)
elif options.boost_network=="Dense":
	model2 = DenseModel(time_step=output_time_step,output_time_step=output_time_step,input_size=output_size,output_size=output_size,lr=0.001)
elif options.boost_network=="Arima":
	model2 = ArimaModel(time_step=output_time_step,output_time_step=output_time_step,input_size=output_size,output_size=output_size,lr=0.001)

#Creating Ensemble Model
if options.ensemble_network is not None:
	model = EnsembleModel(model1,model2,time_step=time_step,input_size=input_size,output_size=output_size,lr=0.001)

# Setting CallBacks
reduce_lr = ReduceLROnPlateau(monitor='loss', factor=0.5, patience=8, min_lr=0.00001,verbose=1)
early_stop=EarlyStopping(monitor='loss', min_delta=0.000005, patience=15, verbose=1)

#Training Ensemble or Base Network
if options.boost_network is None:
	print (model.summary())
	model.fit(X,y,epochs=1000,batch_size=512,callbacks=[reduce_lr,early_stop])

else:
	if len(options.base_network)<8:
		model1.fit(X,y,epochs=1000,batch_size=512,callbacks=[reduce_lr,early_stop])
	pred=model1.predict(X)
	if type(model1.output_shape) is list:
		pred=pred[0]+pred[1]
	model1.trainable=False
	model = BoostingModel(model1,model2,time_step=time_step,input_size=input_size,output_size=output_size,lr=0.001)
	print (model.summary())
	model.fit(X,[y,y-pred],epochs=1000,batch_size=512,callbacks=[reduce_lr,early_stop])

#Saving Weight File for Evaluation
Name=Name.replace(".tsv","")
#Remvoing Strings if Previously saved H5 file
Base=options.base_network.replace("Weights/","")
Base=Base.replace(".h5","")
Base=Base.replace(Name,"")
Name=Name + "-LB-" + str(time_step)
Name=Name + "-FP-" + str(output_time_step)
Name=Name + "-Base-" + Base

if options.ensemble_network is not None:
	Name= Name + "-Ensemble-" + options.ensemble_network
elif options.boost_network is not None:
	Name=Name+"-Boost-"+options.boost_network

model.save("Weights/"+Name+".h5")

out=model.predict(X)

if options.boost_network is not None:
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
evaluate(result,output_size=output_size,Training_Time=0,name="Outputs/Training-"+Name)


