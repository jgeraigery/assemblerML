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

def evaluate(result,output_size=6,Training_Time=0,name=False):
	result2=np.array(result)
	np.save(name+".npy",result2)
	#result[0:Training_Time,output_size*2:output_size*3]=0
	if name!=False:
		plt.figure(1,figsize=(20,10))
		for i in range(output_size):
			plt.subplot(output_size+1, 1, i+1)
			plt.plot(result[:,-1],result[:,i]+result[:,output_size*2+i],c="k",linewidth="4",label="Pred")
			plt.plot(result[:,-1],result[:,output_size+i],c="r",label="True")
			plt.ylabel("State:"+str(i))
			plt.legend(loc="upper right")

		plt.subplot(output_size+1, 1, i+2)
		plt.plot(result[:,-1],result[:,-2],c="r",label="ControlInput")
		plt.ylabel("Control Input")
		plt.xlabel("Time")
		plt.legend(loc="upper right")
		plt.show()
		plt.savefig(name+".svg")
		plt.clf()

		plt.figure(1,figsize=(20,10))
		for i in range(output_size):
			plt.subplot(output_size+1, 1, i+1)
			plt.plot(result[:,-1],result[:,i],c="k",linewidth="4",label="Pred")
			plt.plot(result[:,-1],result[:,output_size+i]-result2[:,output_size*2+i],c="r",label="True")
			plt.ylabel("TD State:"+str(i))
			plt.legend(loc="upper right")

		plt.subplot(output_size+1, 1, i+2)
		plt.plot( result[:,-1],result[:,-2],c="r",label="ControlInput")
		plt.ylabel("Control Input")
		plt.xlabel("Time")
		plt.legend(loc="upper right")
		plt.show()
		plt.savefig(name+"TimeDifference.svg")
		
		plt.clf()


		plt.figure(1,figsize=(20,10))
		for i in range(output_size):
			plt.subplot(output_size+1, 1, i+1)
			plt.plot(result[:200,-1],result[:200,i],c="k",linewidth="4",label="Pred")
			plt.plot(result[:200,-1],result[:200,output_size+i]-result2[:200,output_size*2+i],c="r",label="True")
			plt.ylabel("TD State:"+str(i))
			plt.legend(loc="upper right")

		plt.subplot(output_size+1, 1, i+2)
		plt.plot( result[:200,-1],result[:200,-2],c="r",label="ControlInput")
		plt.ylabel("Control Input")
		plt.xlabel("Time")
		plt.legend(loc="upper right")
		plt.show()
		plt.savefig(name+"TimeDifferenceExploded.svg")
		
		plt.clf()


	for i in range(output_size):
		print ("State:",i," Error: ",np.sqrt(np.mean((result2[Training_Time+1:,output_size+i]-result2[Training_Time+1:,output_size*2+i]-result2[Training_Time+1:,i])**2)))
		print ("State:",i," R2: ",np.corrcoef(result2[Training_Time+1:,output_size+i]-result2[Training_Time+1:,output_size*2+i],result2[Training_Time+1:,i]))

