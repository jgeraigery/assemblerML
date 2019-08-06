# Please change the root variable as required.
# train and test directories of the dataset need to be present in the root directory
# weights need to be present in a weights folder in the root directory

from src.StateSpace import*

import serial
import time

Name="1Input1OutputSS"


time_step=1
output_time_step=1

input_size=3
output_size=2


model = SSModel(time_step=time_step,output_time_step=output_time_step,input_size=input_size,output_size=output_size,lr=0.001)
model.load_weights("1Input1OutputSSRealData.hdf5")


arduino = serial.Serial('COM1', 115200, timeout=.1)
time.sleep(1) #give the connection a second to settle
arduino.write("Hello from Python!")

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
