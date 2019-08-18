#Created By Anand


from sklearn.utils import class_weight
import cv2, numpy as np
from sklearn.model_selection import StratifiedKFold,KFold,train_test_split

import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import cv2, pandas as pd
import numpy as np
import h5py
import pickle, math

from pandas.tools.plotting import autocorrelation_plot
from scipy.stats import probplot as qqplot


from sklearn.metrics import roc_auc_score, accuracy_score
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_squared_error as mse, r2_score as r2



def NormCrossCorr(a,b,mode='same'):
        a = (a - np.mean(a)) / (np.std(a) * len(a))
        b = (b - np.mean(b)) / (np.std(b))
        c = np.correlate(a, b, mode)
        return c

def evaluate(result,name):
	result2=np.array(result)
	result[6:8,:10000]=0
	plt.figure(1)
	plt.subplot(3, 1, 1)
	plt.plot(result[0], result[1]+result[6],c="k",linewidth="4",label="Pred")
	plt.plot( result[0], result[2],c="r",label="True")
	plt.ylabel("State 0")
	plt.subplot(3, 1, 2)
	plt.plot(result[0], result[3]+result[7],c="k",linewidth="4",label="Pred")
	plt.plot(result[0], result[4],c="r",label="True")
	plt.ylabel("State 1")	
	plt.legend(loc="upper right")
	plt.subplot(3, 1, 3)
	plt.plot(result[0], result[5],c="k",label="Input")
	plt.ylabel("Control Input")
	plt.xlabel("Time")
	plt.legend(loc="upper right")
	plt.show()
	plt.savefig(name+".svg")

	result=result2
	plt.clf()

	plt.figure(1)
	plt.subplot(3, 1, 1)
	plt.plot(result[0], result[1],c="k",linewidth="4",label="Pred")
	plt.plot(result[0], result[2]-result[6],c="r",label="True")
	plt.ylabel("TD State 0")
	plt.subplot(3, 1, 2)
	plt.plot(result[0], result[3],c="k",linewidth="4",label="Pred")
	plt.plot(result[0], result[4]-result[7],c="r",label="True")
	plt.ylabel("TD State 1")	
	plt.legend(loc="upper right")
	plt.subplot(3, 1, 3)
	plt.plot(result[0], result[5],c="k",label="Input")
	plt.ylabel("Control Input")
	plt.xlabel("Time")
	plt.legend(loc="upper right")
	plt.show()
	plt.savefig(name+"TimeDifference.svg")
	
	plt.clf()


	plt.figure(1)
	plt.subplot(3, 1, 1)
	plt.plot(result[0,20000:21000], result[1,20000:21000],c="k",linewidth="4",label="Pred")
	plt.plot(result[0,20000:21000], result[2,20000:21000]-result[6,20000:21000],c="r",label="True")
	plt.ylabel("TD State 0")
	plt.subplot(3, 1, 2)
	plt.plot(result[0,20000:21000], result[3,20000:21000],c="k",linewidth="4",label="Pred")
	plt.plot(result[0,20000:21000], result[4,20000:21000]-result[7,20000:21000],c="r",label="True")
	plt.ylabel("TD State 1")	
	plt.legend(loc="upper right")
	plt.subplot(3, 1, 3)
	plt.plot(result[0,20000:21000], result[5,20000:21000],c="k",label="Input")
	plt.ylabel("Control Input")
	plt.xlabel("Time")
	plt.legend(loc="upper right")
	plt.show()
	plt.savefig(name+"TimeDifference20000.svg")
	
	plt.clf()


	#AutoCorrelation Residual Plot State-0
	residuals = ((result[2]-result[6]-result[1])[10001:])
	crosscorr=NormCrossCorr(residuals.flatten(),residuals.flatten(),mode="same")
	plt.plot(np.arange(-20000,20000),crosscorr,c="k",linewidth="4",label="AutoCorrelation-State0")
	plt.ylabel("Correlation Value")
	plt.xlabel("Lag")
	plt.xlim(-1,1)
	plt.legend()
	plt.show()
	plt.savefig("AutoCorrelationState0.svg")
	plt.clf()

	#QQPlot Residuals State-0
	qqplot(residuals.flatten(),plot=plt)
	plt.show()
	plt.savefig("QQState0.svg")
	plt.clf()

	#AutoCorrelation Residual Plot State-1
	
	residuals = ((result[4]-result[3]-result[7])[10001:])
	
	crosscorr=NormCrossCorr(residuals.flatten(),residuals.flatten(),mode="same")
	plt.plot(np.arange(-20000,20000),crosscorr,c="k",linewidth="4",label="AutoCorrelation-State1")
	plt.ylabel("Correlation Value")
	plt.xlabel("Lag")
	plt.xlim(-1,1)
	plt.legend()
	plt.show()
	plt.savefig("AutoCorrelationState1.svg")
	plt.clf()

	#QQPlot Residuals State-1
	
	qqplot(residuals.flatten(),plot=plt)
	plt.show()
	plt.savefig("QQState1.svg")
	plt.clf()
	
	#CrossCorrelation Check
	crosscorr=NormCrossCorr(result[2,10001:]-result[6,10001:],result[1,10001:],mode="same")
	plt.plot(np.arange(-20000,20000),crosscorr,c="k",linewidth="4",label="CrossCorrelation")
	plt.ylabel("Correlation Value")
	plt.xlabel("Lag")
	plt.xlim(-1,1)
	plt.legend()
	plt.show()
	plt.savefig("CrossCorrelationState0.svg")
	plt.clf()

	crosscorr=NormCrossCorr(result[4,10001:]-result[7,10001:],result[3,10001:],mode="same")
	plt.plot(np.arange(-20000,20000),crosscorr,c="k",linewidth="4",label="CrossCorrelation")
	plt.ylabel("Correlation Value")
	plt.xlabel("Lag")
	plt.xlim(-1,1)
	plt.legend()
	plt.show()
	plt.savefig("CrossCorrelationState1.svg")
	plt.clf()


	print "Velocity Error:",np.sqrt(np.mean((result[6,10001:]+result[1,10001:]-result[2,10001:])**2))
	print "Acceleration Error:",np.sqrt(np.mean((result[7,10001:]+result[3,10001:]-result[4,10001:])**2))


	print "Velocity R2:",np.corrcoef(-result[6,10001:]+result[2,10001:],result[1,10001:])
	print "Acceleration R2:",np.corrcoef(result[4,10001:]-result[7,10001:],result[3,10001:])

