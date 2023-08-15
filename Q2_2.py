import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import math
import random

file=pd.read_csv('Dataset.csv')
file = file.to_numpy()
# fixed initilization
centroid=[file[random.randint(0,999)],file[random.randint(0,999)],file[random.randint(0,999)],file[random.randint(0,999)],file[random.randint(0,999)]]
centroid=np.array(centroid)


def closest_centroid_no(x,y,centroid,no_clu):  # returns closest cluster mean number to the given point(x,y)
    j=0
    min=(centroid[j][0]-x)**2+(centroid[j][1]-y)**2
    ans=0
    while(j<no_clu):
        dist=(centroid[j][0]-x)**2+(centroid[j][1]-y)**2
        if(min>dist):
            min=dist
            ans=j
        j+=1
    return(ans)


def vornoi(file,no_clu,centroid):    # Creating vornoi regions
    i=0
    min=[file[0][0],file[0][1]]
    max=[file[0][0],file[0][1]]
    while(i<1000):                  # computing the corners of vornoi region (rectangular)
        if(file[i][0]<min[0]):
            min[0]=file[i][0]
        if(file[i][1]<min[1]):
            min[1]=file[i][1]
        if(file[i][0]>max[0]):
            max[0]=file[i][0]
        if(file[i][1]>max[1]):
            max[1]=file[i][1]
        i+=1

    x=min[0]
    y=min[1]
    while(x<max[0]):         # traversing through all points in rectangle and assigning it color
        y=min[1]             # of cluster to whose mean it is closest
        while(y<max[1]):
            cl_cen=closest_centroid_no(x,y,centroid,no_clu)
            if(cl_cen==0):
                plt.scatter(x,y,color="cyan")
            if(cl_cen==1):
                plt.scatter(x,y,color="orange")
            if(cl_cen==2):
                plt.scatter(x,y,color="blue")
            if(cl_cen==3):
                plt.scatter(x,y,color="green")
            if(cl_cen==4):
                plt.scatter(x,y,color="red")
            y+=0.5
        x+=0.5


def compute_mean(clusters,i):    # computes mean point for given cluster
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

def kmeans(no_clu):

    def distance(centriod,j,point):      # returns distance between mean and given point
        dist=(centroid[j][0]-point[0])**2+(centroid[j][1]-point[1])**2
        dist=math.sqrt(dist)
        return(dist)

    clusters=[[],[],[],[],[]]
    clusters2=[[],[],[],[],[]]
    i=0
    j=0
    clu_no=0
    while(i<1000):
        min=distance(centroid[0],0,file[i])
        clu_no=0
        j=0
        while(j<no_clu):
            if(min>distance(centroid[j],j,file[i])):
                min=distance(centroid[j],j,file[i])
                clu_no=j
            j+=1
        clusters[clu_no].append(file[i])    # append point to cluster whose mean is closest to point
        i+=1

    #copmpute new centroids
    j=0
    while(j<no_clu):
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
        clusters2=[[],[],[],[],[]]
        while(i<no_clu):
            k=0
            while(k<len(clusters[i])):
                min=distance(centroid[i],i,clusters[i][k])
                error_sum+=distance(centroid[i],i,clusters[i][k])
                clu_no=i
                j=0
                while(j<no_clu):
                    if(min>distance(centroid[j],j,clusters[i][k])):
                        min=distance(centroid[j],j,clusters[i][k])
                        clu_no=j
                        nochange=0
                    j+=1
                clusters2[clu_no].append(clusters[i][k])    # append point to cluster whose mean is closest to point
                k+=1
            i+=1


        j=0
        while(j<no_clu):
            mean=compute_mean(clusters2[j],j)
            centroid[j][0]=mean[0]
            centroid[j][1]=mean[1]
            j+=1
        errors.append(error_sum)
        error_sum=0
        clusters.clear()
        clusters=[[],[],[],[],[]]
        i=0
        j=0
        clu_no=0
        nochange=1
        while(i<no_clu):
            k=0
            while(k<len(clusters2[i])):
                min=distance(centroid[i],i,clusters2[i][k])
                error_sum+=distance(centroid[i],i,clusters2[i][k])
                clu_no=i
                j=0
                while(j<no_clu):
                    if(min>distance(centroid[j],j,clusters2[i][k])):
                        min=distance(centroid[j],j,clusters2[i][k])
                        clu_no=j
                        nochange=0
                    j+=1

                clusters[clu_no].append(clusters2[i][k])    # append point to cluster whose mean is closest to point
                k+=1
            i+=1

        # computing new means for clusters
        j=0
        while(j<no_clu):
            mean=compute_mean(clusters[j],j)
            centroid[j][0]=mean[0]
            centroid[j][1]=mean[1]
            j+=1
        errors.append(error_sum)


    i=0
    j=0
    x=[]
    y=[]

    while(j<no_clu):

        while(i<len(clusters[j])):
            x.append(clusters[j][i][0])
            y.append(clusters[j][i][1])
            i+=1                          # plotting clusters
        if(j==0):
            plt.scatter(x,y,color="cyan",label="Cluster 1")
        if(j==1):
             plt.scatter(x,y,color="orange",label="Cluster 2")
        if(j==2):
             plt.scatter(x,y,color="blue",label="Cluster 3")
        if(j==3):
             plt.scatter(x,y,color="green",label="Cluster 4")
        if(j==4):
             plt.scatter(x,y,color="red",label="Cluster 5")
        x.clear()
        y.clear()
        i=0
        j+=1

    plt.xlabel("X-axis")
    plt.ylabel("Y-axis")
    if(no_clu==2):
        plt.title("For K=2")
    if(no_clu==3):
        plt.title("For K=3")
    if(no_clu==4):
        plt.title("For K=4")
    if(no_clu==5):
        plt.title("For K=5")
    plt.legend()
    plt.show()
    vornoi(file,no_clu,centroid)
    plt.xlabel("X-axis")
    plt.ylabel("Y-axis")
    if(no_clu==2):
        plt.title("For K=2")
    if(no_clu==3):
        plt.title("For K=3")
    if(no_clu==4):
        plt.title("For K=4")
    if(no_clu==5):
        plt.title("For K=5")
    plt.show()


i=2
while(i<=5):
    kmeans(i)  # calling k means for k=2 to k=5 times
    i+=1
