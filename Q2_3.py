
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
Imatrix=np.identity(1000,float)
nmatrix=np.empty((1000,1000),float)
i=0
while(i<1000):
    j=0;
    while(j<1000):
        nmatrix[i][j]=1/1000
        j+=1
    i+=1;
# centring the kmatrix
iminusn=Imatrix-nmatrix
kmatrix=np.matmul(iminusn,kmatrix)
kmatrix=np.matmul(kmatrix,iminusn)


#computing eigen value and eigen vectors for kmatrix and sorting them in decreasing order
eig_val,eig_vec=np.linalg.eig(kmatrix)
sort = np.arange(0,len(eig_val), 1)
sort = ([x for _,x in sorted(zip(eig_val, sort))])[::-1]
eig_val = eig_val[sort]
eig_vec = eig_vec[:,sort]

eig_val=eig_val.real
eig_vec=eig_vec.real

top4=eig_vec[:,0:4]    #taking top 4 eigen vectors
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



def compute_mean(clusters,i):    # computing mean for given cluster
    count=len(clusters)
    sumx=0
    sumy=0
    j=count-1
    while(j>=0):
        sumx+=clusters[j][0]
        sumy+=clusters[j][1]
        j-=1
    mean=[sumx/count,sumy/count]
    return(mean)


def kmeans(top4):
    pointmap=np.empty((1000,1),int)
    file=top4
    file = np.array(top4)
    centroid=[file[random.randint(0,999)],file[random.randint(0,999)],file[random.randint(0,999)],file[random.randint(0,999)]]
    centroid=np.array(centroid)
    def distance(centriod,j,point):
        dist=(centroid[j][0]-point[0])**2+(centroid[j][1]-point[1])**2+(centroid[j][2]-point[2])**2+(centroid[j][3]-point[3])**2
        dist=math.sqrt(dist)
        return(dist)

    clusters=[[],[],[],[]]
    clusters2=[[],[],[],[]]
    i=0
    j=0
    clu_no=0
    while(i<1000):
        min=distance(centroid[0],0,file[i])
        clu_no=0
        j=0
        while(j<4):
            if(min>distance(centroid[j],j,file[i])):
                min=distance(centroid[j],j,file[i])
                clu_no=j
            j+=1
        clusters[clu_no].append(file[i])   # append point to cluster whose mean is closest to point
        pointmap[i]=clu_no
        i+=1

    #copmpute new centroids
    j=0
    while(j<4):
        mean=compute_mean(clusters[j],j)
        centroid[j][0]=mean[0]
        centroid[j][1]=mean[1]
        j+=1
    nochange=0
    error_sum=0
    errors=[]
    while(nochange==0):
        i=0
        j=0
        clu_no=0
        error_sum=0
        nochange=1
        clusters2.clear()
        clusters2=[[],[],[],[]]
        while(i<4):
            k=0
            while(k<len(clusters[i])):
                min=distance(centroid[i],i,clusters[i][k])
                error_sum+=distance(centroid[i],i,clusters[i][k])
                clu_no=i
                j=0
                while(j<4):
                    if(min>distance(centroid[j],j,clusters[i][k])):
                        min=distance(centroid[j],j,clusters[i][k])
                        clu_no=j
                        nochange=0
                    j+=1
                clusters2[clu_no].append(clusters[i][k])      # append point to cluster whose mean is closest to point
                pointmap[i]=clu_no
                k+=1
            i+=1


        j=0
        while(j<4):
            mean=compute_mean(clusters2[j],j)
            centroid[j][0]=mean[0]
            centroid[j][1]=mean[1]
            j+=1
        errors.append(error_sum)
        error_sum=0
        clusters.clear()
        clusters=[[],[],[],[]]
        i=0
        j=0
        clu_no=0
        nochange=1
        while(i<4):
            k=0
            while(k<len(clusters2[i])):
                min=distance(centroid[i],i,clusters2[i][k])
                error_sum+=distance(centroid[i],i,clusters2[i][k])
                clu_no=i
                j=0
                while(j<4):
                    if(min>distance(centroid[j],j,clusters2[i][k])):
                        min=distance(centroid[j],j,clusters2[i][k])
                        clu_no=j
                        nochange=0
                    j+=1

                clusters[clu_no].append(clusters2[i][k])      # append point to cluster whose mean is closest to point
                pointmap[i]=clu_no
                k+=1
            i+=1


        j=0
        while(j<4):
            mean=compute_mean(clusters[j],j)
            centroid[j][0]=mean[0]
            centroid[j][1]=mean[1]
            j+=1
        errors.append(error_sum)

    i=0
    x=[]
    y=[]
    file2=pd.read_csv('Dataset.csv')
    file2= file2.to_numpy()
    while(i<len(file2)):
        if(pointmap[i]==0):
            x.append(file2[i][0])
            y.append(file2[i][1])
        i+=1
    plt.scatter(x,y,color="maroon",label="Cluster 1")
    x.clear()
    y.clear()
    i=0
    while(i<len(file2)):
        if(pointmap[i]==1):
            x.append(file2[i][0])
            y.append(file2[i][1])
        i+=1
    plt.scatter(x,y,color="green",label="Cluster 2")
    x.clear()
    y.clear()
    i=0
    while(i<len(file2)):
        if(pointmap[i]==2):
            x.append(file2[i][0])
            y.append(file2[i][1])
        i+=1
    plt.scatter(x,y,color="orange",label="Cluster 3")
    x.clear()
    y.clear()
    i=0
    while(i<len(file2)):
        if(pointmap[i]==3):
            x.append(file2[i][0])
            y.append(file2[i][1])
        i+=1
    plt.scatter(x,y,color="blue",label="Cluster 4")
    plt.xlabel("X-axis")
    plt.ylabel("Y-axis")
    plt.title("Spectral Clustring using polynomial map d=2")
    plt.legend()
    plt.show()


kmeans(top4)
