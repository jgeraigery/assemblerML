# Please change the root variable as required.
# train and test directories of the dataset need to be present in the root directory
# weights need to be present in a weights folder in the root directory

from Problems.DCMotor import* 
from Models.DenseModel import* 
from Operators.Ensemble import* 

m = Motor()
dT = .001
m.setTimeStep(dT)
model = rnnmodel()
result = np.zeros([8,1])
printDuring=False

X=[]
y=[]
movingvalue=np.zeros((3,3))

for i in np.arange(0, 50000):
    # control input function
    if ( np.mod(i,50) == 0 ):
        print ("Loop-",i)
	controlInput=0
	if (i%1000==0 and i!=0 and i<30000):
        	controlInput = 10
	elif (i%1000==0 and i!=0 and i>30000):
        	controlInput = getControlInput()
    if (i%10000==0):
	#m.J=m.J+0.1
	m.update()
    stateTensor =(m.state)
    stateTensor = np.concatenate((stateTensor,(np.ones([1,1], dtype=float) * controlInput)), 1)
    outBar=m.step(controlInput)
    movingvalue[0:2,:]=movingvalue[1:3,:]
    movingvalue[2,:]=stateTensor
    movingvalue2=np.asarray(list(movingvalue))

    if i<10000:
        out=np.zeros((1,2))
        X.append(movingvalue2)
        y.append(outBar-stateTensor[:,0:2])

    elif i==10000:
        out=np.zeros((1,2))
        model.fit(np.asarray(X),(np.asarray(y)).reshape(10000,2),epochs=50)
	adam=keras.optimizers.Adam(lr=0.00001, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False)
        model.compile(loss="mse", optimizer=adam, metrics=['accuracy'])
    elif i>10000:
    	out=model.predict(movingvalue2.reshape(1,3,3))
    	model.fit(movingvalue2.reshape(1,3,3),outBar-stateTensor[:,0:2],epochs=1)
    else:
	continue
    tmpResult = np.empty([8,1])
    tmpResult[0] = dT*(i+1)
    tmpResult[1] = out[0][0]
    tmpResult[2] = outBar[0][0]
    tmpResult[3] = out[0][1]
    tmpResult[4] = outBar[0][1]
    tmpResult[5] = controlInput
    tmpResult[6] = stateTensor[0][0]
    tmpResult[7] = stateTensor[0][1]
    result = np.concatenate((result,tmpResult),1)


evaluate(result,"RNNArima")

print model.get_weights()
