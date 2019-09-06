import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import h5py
import pickle,math

from optparse import OptionParser

#Writing Parser for Python File
parser = OptionParser()

parser.add_option("-p", "--path", action="append",dest="training_path", help="Path to training data in the form of numpy array")
(options, args) = parser.parse_args()

if not options.training_path:   # if filename is not given
        parser.error('Error: path to training data must be specified. Pass --path to command line')


#python BBPlot.py -p Testing-0.1Torque_20HZ_DataPhysics.npy -p Testing-0.1Torque_20HZ_DataPhysics-Base-0.1Torque_20HZ_Data.npy-LB-1-FP-1-Base-Physics-Boost-Dense.npy -p Testing-0.1Torque_20HZ_DataPhysics-Base-0.1Torque_20HZ_Data.npy-LB-1-FP-1-Base-Physics-Boost-Linear.npy -p Testing-0.1Torque_20HZ_DataPhysics-Base-0.1Torque_20HZ_Data.npy-LB-1-FP-1-Base-Physics-Ensemble-Dense.npy -p Testing-0.1Torque_20HZ_DataPhysics-Base-0.1Torque_20HZ_Data.npy-LB-10-FP-1-Base-Physics-Boost-RNN.npy -p Testing-0.1Torque_20HZ_Data-Base-.npy-LB-1-FP-1-Base-Linear.npy -p Testing-0.1Torque_20HZ_Data-Base-.npy-LB-10-FP-1-Base-RNN.npy -p Testing-0.1Torque_20HZ_Data-Base-.npy-LB-1-FP-1-Base-Linear-Boost-Dense.npy -p Testing-0.1Torque_20HZ_Data-Base-.npy-LB-10-FP-1-Base-Dense.npy

Name=options.training_path
label=np.asarray(options.training_path)

for i in range(len(label)):
	label[i]=label[i].replace("Testing-2Torque_20HZ_DataPhysics-Base-0.1Torque_20HZ_Data.npy-LB-1-FP-1-Base-","")
	label[i]=label[i].replace("Testing-2Torque_20HZ_Data-Base-0.1Torque_20HZ_Data.npy-LB-1-FP-1-Base-","")
	label[i]=label[i].replace("Testing-2Torque_20HZ_Data-Base-0.1Torque_20HZ_Data.npy-LB-10-FP-1-Base-","")
	label[i]=label[i].replace("Testing-2Torque_20HZ_DataPhysics-Base-0.1Torque_20HZ_Data.npy-LB-10-FP-1-Base-","")
	label[i]=label[i].replace("Testing-2Torque_20HZ_Data","")

	label[i]=label[i].replace("Testing-0.1Torque_20HZ_DataPhysics-Base-0.1Torque_20HZ_Data.npy-LB-1-FP-1-Base-","")
	label[i]=label[i].replace("Testing-0.1Torque_20HZ_DataPhysics-Base-0.1Torque_20HZ_Data.npy-LB-10-FP-1-Base-","")
	label[i]=label[i].replace("Testing-0.1Torque_20HZ_Data-Base-.npy-LB-1-FP-1-Base-","")
	label[i]=label[i].replace("Testing-0.1Torque_20HZ_Data-Base-.npy-LB-10-FP-1-Base-","")
	label[i]=label[i].replace("Testing-0.1Torque_20HZ_Data","")

	label[i]=label[i].replace("Training-0.1Torque_20HZ_Data-LB-1-FP-1-Base-","")
	label[i]=label[i].replace("Training-0.1Torque_20HZ_Data.npy-LB-1-FP-1-Base-","")


	label[i]=label[i].replace(".npy","")

print (label,len(label))

output_size=3

red_square = dict(markerfacecolor='r', marker='s')

plt.figure(1,figsize=(30,10))
for i in range(3):
	error=[]
	for name in Name:
		x=np.load(name)
		#error.append(np.sqrt(np.square(x[:,output_size+i]-x[:,i]-x[:,output_size*2+i])))
		error.append(np.array(np.corrcoef(x[:,output_size+i]-x[:,i],x[:,output_size*2+i])[0][1]))
	error=np.asarray(error)
	error=np.reshape(error,(1,12))
	plt.subplot(output_size,1,i+1)
	plt.boxplot(error,flierprops=red_square,labels=label,showmeans=True)
	#plt.plot(error,label=label)
	#plt.ylim(0.94,1)
	plt.ylabel("Correlation for State:"+str(i))
	#plt.legend(loc="upper right")

plt.show()
plt.savefig("CorrelationInterpolation.svg")
plt.clf()


