"""
First attempt at closing the control loop with DC motor statespace/boosted modeling.
System architecture:

Motor -> enocder position/input voltage/state readings -> NIDAQMX data logger -> run through boosted SS model ->
calculate next output -> write to arduino in form: Q±#####\n

Task timing is important.  Need to measure loop timings.

8-6-2019, Anand Ramakrishnan, Kent Evans

"""

# Please change the root variable as required.
# train and test directories of the dataset need to be present in the root directory
# weights need to be present in a weights folder in the root directory

from src.StateSpace import*

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

ard = serial.Serial(COMPORT, 115200, timeout=.1)
#time.sleep(1) #give the connection a second to settle
# arduino.write("Hello from Python!")

#Measurement Configuration
encoderTask = nidaqmx.Task()
encoderChan = encoderTask.ci_channels.add_ci_ang_encoder_chan("Dev1/ctr0",pulses_per_rev=300)
encoderChan.ci_encoder_a_input_term("Dev1/PF8")
encoderChan.ci_encoder_b_input_term("Dev1/PF10")

inputTask = nidaqmx.Task()
inputChan = inputTask.ai_channels.add_ai_voltage_chan("Dev1/ai4")
inputTask.timing.samp_clk_rate(20000)

inputTask.start()
encoderTask.start()
lastPos = 0
datArray = np.zeros(1,1,3)
state = np.zeros(1,1,3)

currTime = time.clock()
while True:
    state[3] = inputTask.read(number_of_samples_per_channel=40)
    state[0] = encoderTask.read(number_of_samples_per_channel=1)
    state[1] = state[0]-lastPos
    lastPos = state[0]
    np.append(datArray,state,1)
    while time.clock() < currTime + .002:
        pass


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