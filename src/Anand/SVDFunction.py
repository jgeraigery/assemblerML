#Looks at singular value decomposition
import numpy as np
import matplotlib.pyplot as plt
import math as mt

Npr=2**2    #number of rows
Npc=2**4   #number of cols
#a=np.random.rand(Np,Np)
a=np.random.normal(0,1,[Npr,Npc])
Npmx=max(Npr,Npc)
Npmn=min(Npr,Npc)
Np=30
#a=np.zeros([Npr,Npc])
for i in range(Npmn):
    a[i,i]=(Np-i)*4
#print(a)
u, s, vh = np.linalg.svd(a, full_matrices=True)
#function of s
sv=np.zeros([Npr,Npc])
#vh=np.transpose(vh)
for i in range(Npmn):
    sv[i,i]= mt.exp(s[i])
aprime=np.transpose(u)@sv@vh
adiff=a-aprime
# u.shape, s.shape, vh.shape
sinv=np.zeros([Npc,Npr])
#vh=np.transpose(vh)
for i in range(Npmn):
    sinv[i,i]=mt.log(s[i])
ainv=np.transpose(vh)@sinv@u

indt1=a@ainv
indt2=aprime@ainv
indt3=ainv@aprime
vi=np.transpose(vh)@vh
ui=u@np.transpose(u)
#print(u)
#print(s)
#print(vh)



#mu, sigma = 100, 15vh
#x = mu + sigma * np.random.randn(10000)
hist, bins = np.histogram(s, bins=int(Np/3))
width = 0.7 * (bins[1] - bins[0])
center = (bins[:-1] + bins[1:]) / 2
plt.bar(center, hist, align='center', width=width)
plt.show()