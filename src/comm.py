"""
First attempt at closing the control loop with DC motor statespace/boosted modeling.
System architecture:

Motor -> enocder position/input voltage/state readings -> NIDAQMX data logger -> run through boosted SS model ->
calculate next output -> write to arduino in form:

Task timing is important.  Need to measure loop timings.

8-6-2019, Anand Ramakrishnan, Kent Evans

"""

# Please change the root variable as required.
# train and test directories of the dataset need to be present in the root directory
# weights need to be present in a weights folder in the root directory
import keras_preprocessing
from StateSpace import *

import nidaqmx
import serial
import time
import numpy as np

Name="1Input1OutputSS"
COMPORT = 'COM11'

time_step=1
output_time_step=1

input_size=3
output_size=2

model = SSModel(time_step=time_step,output_time_step=output_time_step,input_size=input_size,output_size=output_size,lr=0.001)
model.load_weights("1Input1OutputSSRealData.hdf5")

ard = serial.Serial(COMPORT, 52600, timeout=.1)
#time.sleep(1) #give the connection a second to settle
# arduino.write("Hello from Python!")

#Measurement Configuration
encoderTask = nidaqmx.Task()
encoderChan = encoderTask.ci_channels.add_ci_ang_encoder_chan("Dev1/ctr0",pulses_per_rev=300)
encoderChan.ci_encoder_a_input_term="/Dev1/PFI8"
encoderChan.ci_encoder_b_input_term="/Dev1/PFI10"
print(encoderChan.ci_encoder_a_input_term)
print(encoderChan.ci_encoder_b_input_term)

inputTask = nidaqmx.Task()
inputChan = inputTask.ai_channels.add_ai_voltage_chan("Dev1/ai4")
inputTask.timing.samp_clk_rate=20000
#print(inputTask.timing.samp_clk_rate)

inputTask.start()
encoderTask.start()
lastPos = 0
datArray = np.zeros((1,1,3))
state = np.zeros((1,1,3))

currTime = time.clock()
startTime = time.clock()
print(currTime)
true=True
incr=1
while true:
    if incr % 1==0:
        print(datArray.shape)
        pred = model.predict(datArray[[incr-1],:,:])
        pred = datArray[incr-1, :, 0:2] + (pred/50)
        print("Pred,State", pred, state)
    incr+=1
    raw = inputTask.read(number_of_samples_per_channel=40)
    state[:,:,2] = np.mean(raw)
    state[:,:,0] = encoderTask.read(number_of_samples_per_channel=1)
    state[:,:,1] = (state[:,:,0]-lastPos)/(2.5*10-3)
    lastPos = np.array(state[:,:,0])
    datArray=np.append(datArray,state,0)
    while time.clock() < currTime + .002:
        pass
    currTime = time.clock()
    if currTime-startTime>10:
        true=False
"""
while True:
	try:
		data = arduino.readline()
		try:
			decoded_data= data.rstrip('\n')
			decoded_data= float(decoded_data.decode("utf-8"))
			decoded_data=np.reshape(np.asarray(decoded_data),(1,1,3))
    		pred=model.predict(decoded_data)
			arduino.write(str(pred))
		except:
			print("Error in Decoding Data")
			continue
	except:
		print("Error in Reading Data")
"""

ard.close()
