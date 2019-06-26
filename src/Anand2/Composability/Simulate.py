# Please change the root variable as required.
# train and test directories of the dataset need to be present in the root directory
# weights need to be present in a weights folder in the root directory


from Problems.DCMotor import* 
from Problems.InvertedPendulum import* 
from Models.Dense import* 
from Models.StateSpace import* 
from Models.RNN import* 
from Operators.Ensemble import* 
from Operators.Boosting import* 
from Evaluation.Evaluate import* 

m = Inverted_Pendulum()
dT = .0001
m.setTimeStep(dT)

time_step=1
output_time_step=1

input_size=m.input_size
output_size=m.output_size

simulation_time=50000
training_time=10000

train_on_fly=True

model1 = RNNModel(time_step=time_step,output_time_step=output_time_step,input_size=input_size,output_size=output_size)
model2 = RNNModel(time_step=time_step,output_time_step=output_time_step,input_size=input_size,output_size=output_size)

model= EnsembleModel(model1,model2,time_step=time_step,input_size=input_size,output_size=output_size)

result = np.zeros([1,output_size*3+2])

X=[]
y=[]

moving_input=np.zeros((time_step,input_size))
moving_output=np.zeros((output_time_step,output_size))

for i in np.arange(0,simulation_time):
    # control input function
	if (i%100==0):
		print "Simulation Time:",i*dT
		controlInput = m.getControlInput()

	#if (i%10000==0):
		#m.J=m.J+0.1
	#	m.update()

	stateTensor =(m.state)
	stateTensor = np.concatenate((stateTensor,(np.ones([1,1], dtype=float) * controlInput)), 1)
	outBar=m.step(controlInput)


	if time_step>1: #To shift Inputs to the left
		moving_input[0:time_step-1,:]=moving_input[1:time_step,:]

	moving_input[time_step-1,:]=stateTensor
	moving_input2=np.asarray(list(moving_input))

	if output_time_step>1:
		moving_output[0:output_time_step-1,:]=moving_output[1:output_time_step,:]

	moving_output[output_time_step-1,:]=outBar-stateTensor[:,0:output_size]
	moving_output2=np.asarray(list(moving_output))


	if i<training_time:

		out=np.zeros((1,output_time_step,output_size))
		X.append(moving_input2)
		y.append(moving_output2)

	elif i==training_time:
		out=np.zeros((1,output_time_step,output_size))
		model.fit(np.asarray(X),np.asarray(y),epochs=50)
		if train_on_fly:
			adam=keras.optimizers.Adam(lr=0.00001, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False)
			model.compile(loss="mse", optimizer=adam, metrics=['accuracy'])

	elif i>training_time:
		out=model.predict(moving_input2.reshape(1,time_step,input_size))
		if train_on_fly:
			model.fit(moving_input2.reshape(1,time_step,input_size),(moving_output2).reshape(1,output_time_step,output_size),epochs=1)
	else:
		continue

	tmpResult = np.empty([1,output_size*3+2])
	tmpResult[:,0:output_size] = out.reshape(output_size)
	tmpResult[:,output_size:output_size*2] = outBar
	tmpResult[:,output_size*2:output_size*3+1] = stateTensor
	tmpResult[:,-1] = dT*(i+1)
	result = np.concatenate((result,tmpResult),0)


evaluate(result,output_size=output_size,Training_Time=training_time,name="Images/DenseEnsembleIP")

