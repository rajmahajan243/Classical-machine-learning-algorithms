# Q1.3_B Kernel PCA
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def kernelPCA(sigma):
    file=pd.read_csv('Dataset.csv')
    i=0;
    filet=file.T
    file = file.to_numpy()
    filet = filet.to_numpy()
    kmatrix=np.empty((1000,1000),float)
    while(i<1000):
        j=0;
        while(j<1000):
            kmatrix[i][j]=np.exp(-1*(file[i]-file[j]).T@(file[i]-file[j])/(2*sigma))
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

    iminusn=Imatrix-nmatrix
    kmatrix=np.matmul(iminusn,kmatrix)
    kmatrix=np.matmul(kmatrix,iminusn)

    centred_data=kmatrix
    centred_data
    eig_val,eig_vec=np.linalg.eig(centred_data)
    sort = np.arange(0,len(eig_val), 1)
    sort = ([x for _,x in sorted(zip(eig_val, sort))])[::-1]
    eig_val = eig_val[sort]
    eig_vec = eig_vec[:,sort]

    eig_val=eig_val.real
    eig_vec=eig_vec.real


    lamda=(np.abs(eig_val))**0.5
    eig_vec=eig_vec/lamda

    top2=eig_vec[:,0:2]
    k_components=centred_data.T@top2
    np.array(k_components)

    graph(k_components,sigma)


def graph(k_components,sigma):
    if(sigma==0.1):
        plt.scatter(k_components[:,0],k_components[:,1])
        plt.title("Projection of points using top 2 PCs for sigma=0.1")
        plt.xlabel("Principal Component 1")
        plt.ylabel("Principal Component 2")
        plt.show()
    if(sigma==0.2):
        plt.scatter(k_components[:,0],k_components[:,1])
        plt.title("Projection of points using top 2 PCs for sigma=0.2")
        plt.xlabel("Principal Component 1")
        plt.ylabel("Principal Component 2")
        plt.show()
    if(sigma==0.3):
        plt.scatter(k_components[:,0],k_components[:,1])
        plt.title("Projection of points using top 2 PCs for sigma=0.3")
        plt.xlabel("Principal Component 1")
        plt.ylabel("Principal Component 2")
        plt.show()
    if(sigma==0.4):
        plt.scatter(k_components[:,0],k_components[:,1])
        plt.title("Projection of points using top 2 PCs for sigma=0.4")
        plt.xlabel("Principal Component 1")
        plt.ylabel("Principal Component 2")
        plt.show()
    if(sigma==0.5):
        plt.scatter(k_components[:,0],k_components[:,1])
        plt.title("Projection of points using top 2 PCs for sigma=0.5")
        plt.xlabel("Principal Component 1")
        plt.ylabel("Principal Component 2")
        plt.show()
    if(sigma==0.6):
        plt.scatter(k_components[:,0],k_components[:,1])
        plt.title("Projection of points using top 2 PCs for sigma=0.6")
        plt.xlabel("Principal Component 1")
        plt.ylabel("Principal Component 2")
        plt.show()
    if(sigma==0.7):
        plt.scatter(k_components[:,0],k_components[:,1])
        plt.title("Projection of points using top 2 PCs for sigma=0.7")
        plt.xlabel("Principal Component 1")
        plt.ylabel("Principal Component 2")
        plt.show()
    if(sigma==0.8):
        plt.scatter(k_components[:,0],k_components[:,1])
        plt.title("Projection of points using top 2 PCs for sigma=0.8")
        plt.xlabel("Principal Component 1")
        plt.ylabel("Principal Component 2")
        plt.show()
    if(sigma==0.9):
        plt.scatter(k_components[:,0],k_components[:,1])
        plt.title("Projection of points using top 2 PCs for sigma=0.9")
        plt.xlabel("Principal Component 1")
        plt.ylabel("Principal Component 2")
        plt.show()
    if(sigma==1):
        plt.scatter(k_components[:,0],k_components[:,1])
        plt.title("Projection of points using top 2 PCs for sigma=1")
        plt.xlabel("Principal Component 1")
        plt.ylabel("Principal Component 2")
        plt.show()


i=0.1
while(i<=1):
    i=round(i,1)
    kernelPCA(i)
    i+=0.1    
