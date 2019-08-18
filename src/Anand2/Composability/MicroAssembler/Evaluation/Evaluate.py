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
		plt.subplot(1, 1, 1)
		plt.plot(result[:,0]+result[:,output_size*2+0],result[:,1]+result[:,output_size*2+1],c="k",linewidth="4",label="Pred")
		plt.plot(result[:,output_size+0],result[:,output_size+1],c="r",label="True")
		plt.ylabel("Position-Y")
		plt.xlabel("Position-X")
		plt.legend(loc="upper right")
		plt.show()
		plt.savefig(name+".svg")
		plt.clf()

		plt.figure(1,figsize=(20,10))
		plt.subplot(1, 1, 1)
		plt.plot(result[:,0],result[:,1],c="k",linewidth="4",label="Pred")
		plt.plot(result[:,output_size+0]-result2[:,output_size*2+0],result[:,output_size+1]-result2[:,output_size*2+1],c="r",label="True")
		plt.ylabel("TD Position-Y")
		plt.xlabel("TD Position-X")
		plt.legend(loc="upper right")
		plt.show()

		plt.savefig(name+"TimeDifference.svg")
		
		plt.clf()


		plt.figure(1,figsize=(20,10))
		plt.subplot(1, 1, 1)
		plt.plot(result[0:200,0],result[0:200,1],c="k",linewidth="4",label="Pred")
		plt.plot(result[0:200,output_size+0]-result2[0:200,output_size*2+0],result[0:200,output_size+1]-result2[0:200,output_size*2+1],c="r",label="True")
		plt.ylabel("TD Position-Y")
		plt.xlabel("TD Position-X")
		plt.legend(loc="upper right")
		plt.show()

		plt.savefig(name+"TimeDifferenceExploded.svg")
		
		plt.clf()


