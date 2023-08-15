import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import math
import random






def compute_mean(clusters,i):     # Function to compute mean
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

def kmeans(initilization):
    file=pd.read_csv('Dataset.csv')
    file = file.to_numpy()
    # Initializing random means for clusters
    centroid=[file[random.randint(0,999)],file[random.randint(0,999)],file[random.randint(0,999)],file[random.randint(0,999)]]
    centroid=np.array(centroid)
    def distance(centriod,j,point):   # computes distance between mean of cluster and given point
        dist=(centroid[j][0]-point[0])**2+(centroid[j][1]-point[1])**2
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
                clusters2[clu_no].append(clusters[i][k])  # append point to cluster whose mean is closest to point
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

                clusters[clu_no].append(clusters2[i][k])    # append point to cluster whose mean is closest to point
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
    plt.scatter(x,y,color="cyan",label="Cluster 2")
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
    if(initilization==1):
        plt.title("Case 1")
    if(initilization==2):
        plt.title("Case 2")
    if(initilization==3):
        plt.title("Case 3")
    if(initilization==4):
        plt.title("Case 4")
    if(initilization==5):
        plt.title("Case 5")
    plt.legend()
    plt.show()

    iterations=[]
    i=0
    while(i<len(errors)):
        iterations.append(i)
        i+=1
    plt.plot(iterations,errors,color='green')
    plt.ylabel("Error")
    plt.xlabel("Number of Iterations")
    plt.title("Error Function")
    plt.show()


i=1
while(i<=5):
    kmeans(i)
    i+=1
