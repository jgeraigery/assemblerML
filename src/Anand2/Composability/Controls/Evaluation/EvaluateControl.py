#Created By Anand



import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import h5py
import pickle, math

#from sklearn.metrics import roc_auc_score, accuracy_score
#from sklearn.preprocessing import LabelEncoder
#from sklearn.metrics import mean_squared_error as mse, r2_score as r2



def NormCrossCorr(a,b,mode='same'):
	a = (a - np.mean(a)) / (np.std(a) * len(a))
	b = (b - np.mean(b)) / (np.std(b))
	c = np.correlate(a, b, mode)
	return c

def evaluate(result,output_size=2,Training_Time=0000,name=False):
	if name!=False:
		plt.figure(1,figsize=(20,10))
		for i in range(output_size):
			plt.subplot(output_size+1, 1, i+1)
			plt.plot(result[:,i],c="k",linewidth="4",label="CurrentState")
			plt.plot(result[:,output_size+i],c="r",label="ReferenceState")
			plt.ylabel("State:"+str(i))
			plt.legend(loc="upper right")

		plt.subplot(output_size+1, 1, i+2)
		plt.plot(result[:,-1],c="r",label="ControlInput")
		plt.ylabel("Control Input")
		plt.xlabel("Time")
		plt.legend(loc="upper right")
		plt.show()
		plt.savefig(name+".svg")
		plt.clf()

		plt.figure(1,figsize=(20,10))
		for i in range(output_size):
			plt.subplot(output_size+1, 1, i+1)
			plt.plot(result[500:600,i],c="k",linewidth="4",label="CurrentState")
			plt.plot(result[500:600,output_size+i],c="r",label="ReferenceState")
			plt.ylabel("State:"+str(i))
			plt.legend(loc="upper right")

		plt.subplot(output_size+1, 1, i+2)
		plt.plot(result[500:600,-1],c="r",label="ControlInput")
		plt.ylabel("Control Input")
		plt.xlabel("Time")
		plt.legend(loc="upper right")
		plt.show()
		plt.savefig(name+"ExplodedView.svg")
		plt.clf()

