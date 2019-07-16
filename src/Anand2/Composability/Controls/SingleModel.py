# Please change the root variable as required.
# train and test directories of the dataset need to be present in the root directory
# weights need to be present in a weights folder in the root directory

from Problems.DCMotor import *
from numpy import sin,cos,pi
from Models.SingleDense import* 
from Models.Dense import* 
from Models.StateSpace import* 
from Models.RNN import* 
from Operators.Ensemble import* 
from Operators.Boosting import* 
from Operators.CyclicControl import* 
from Operators.SingleControl import* 
from Evaluation.Evaluate import* 
import pandas as pd
Name="ControllerTrial"
import time

m = Motor()

dT=0.1
m.setTimeStep(dT)
time_step=1
output_time_step=1

input_size=3
output_size=2

model1 = SingleDenseModel(time_step=time_step,output_time_step=output_time_step,input_size=5,output_size=2)

model=SingleControlModel(model1)
print (model.summary())

m.reset()
stateTensor=m.state
stateTensor=stateTensor.reshape(1,1,2)
ref=np.array([0.1,0.1])
ref=ref.reshape(1,1,2)
controlInput=np.zeros((1,1,1))
stateNew=np.array(stateTensor)
X=[]
y=[]
pred=0
for i in np.arange(0,25000):
	print ("Reference,Predicted State, True State,Control Input,Loop Count",ref,pred,stateTensor,controlInput[0][0][0],i)
	time.sleep(0.1)

	#X.append([stateTensor,ref-stateTensor,controlInput])
	#y.append([ref[:,:,:],stateTensor])
	pred,controlInput=model.predict([stateTensor,ref,controlInput])
	stateTensorNew=m.step(controlInput[0][0][0])
	stateTensorNew=stateTensorNew.reshape(1,1,2)
	print (controlInput,stateTensorNew,pred)
	model.fit([stateTensor,ref,controlInput],[stateTensorNew,controlInput],epochs=10,verbose=1)
	stateTensor=stateTensorNew


