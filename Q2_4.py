
import math
import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt



file=pd.read_csv('Dataset.csv')

i=0;
file = file.to_numpy()
kmatrix=np.empty((1000,1000),float)
while(i<1000):
    j=0;
    while(j<1000):
        kmatrix[i][j]=(1+np.dot(file[i],file[j]))**2
        j+=1

    i+=1;


eig_val,eig_vec=np.linalg.eig(kmatrix)
sort = np.arange(0,len(eig_val), 1)
sort = ([x for _,x in sorted(zip(eig_val, sort))])[::-1]
eig_val = eig_val[sort]
eig_vec = eig_vec[:,sort]

eig_val=eig_val.real
eig_vec=eig_vec.real

top4=eig_vec[:,0:4]
i=0
sum=0
while(i<1000):
    sum=0
    j=0
    while(j<4):
        sum+=top4[i][j]**2
        j+=1
    top4[i]=top4[i]/sum
    i+=1

def argmax(vec):
    i=0
    max=vec[0]
    ans=0
    while(i<4):
        if(max<vec[i]):
            max=vec[i]
            ans=i
        i+=1

    return(ans)


clusters=[[],[],[],[]]
i=0
while(i<1000):
    r=argmax(top4[i])
    clusters[r].append(file[i])
    i+=1

i=0
x=[]
y=[]
while(i<len(clusters[0])):
    x.append(clusters[0][i][0])
    y.append(clusters[0][i][1])
    i+=1
plt.scatter(x,y,color="maroon",label="Cluster 1")
x.clear()
y.clear()
i=0
while(i<len(clusters[1])):
    x.append(clusters[1][i][0])
    y.append(clusters[1][i][1])
    i+=1
plt.scatter(x,y,color="green",label="Cluster 2")
x.clear()
y.clear()
i=0
while(i<len(clusters[2])):
    x.append(clusters[2][i][0])
    y.append(clusters[2][i][1])
    i+=1
plt.scatter(x,y,color="orange",label="Cluster 3")
x.clear()
y.clear()
i=0
while(i<len(clusters[3])):
    x.append(clusters[3][i][0])
    y.append(clusters[3][i][1])
    i+=1
plt.scatter(x,y,color="blue",label="Cluster 4")
plt.xlabel("X-axis")
plt.ylabel("Y-axis")
plt.title("Clustring")
plt.legend()
plt.show()
