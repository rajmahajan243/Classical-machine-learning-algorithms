# Q1.1  With centring
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

file=pd.read_csv('Dataset.csv')

mean=file.mean()
#centring the Data
centred_data=file-mean

#compute covarience
cov=np.cov(centred_data.T)


# find eigen vectors and eigen values of covarience matrix
eig_val,eig_vec=np.linalg.eig(cov)
sort = np.arange(0,len(eig_val), 1)
sort = ([x for _,x in sorted(zip(eig_val, sort))])[::-1]
eig_val = eig_val[sort]
eig_vec = eig_vec[:,sort]


# varience explained by each principal component
sum1=sum(eig_val)
list1=["pc1 %","pc2 %"]
list2=[eig_val[0]*100/sum1,eig_val[1]*100/sum1]
print("Eigen Value 1 = ",eig_val[0])
print("Eigen Value 2 = ",eig_val[1])
print("Percentage of variance contributed by principal component 1 = ", eig_val[0]*100/sum1)
print("Percentage of variance contributed by principal component 2 = ",eig_val[1]*100/sum1)


plt.scatter(centred_data.iloc[:,0],centred_data.iloc[:,1])
plt.xlabel("X-axis")
plt.ylabel("Y-axis")
plt.title("Given Data points after centring")
plt.show()
plt.scatter(file.iloc[:,0],file.iloc[:,1],color="white")
listx=[0,eig_vec[0][0]]
listy=[0,eig_vec[0][1]]
plt.axline(eig_vec[0],slope=eig_vec[0][1]/eig_vec[0][0],color='red',label="PC1")
listx=[0,eig_vec[1][0]]
listy=[0,eig_vec[1][1]]
plt.axline(eig_vec[1],slope=eig_vec[1][1]/eig_vec[1][0],color='cyan',label="PC2")
plt.xlabel("X-axis")
plt.ylabel("Y-axis")
plt.title("Top two Principal Components after centring")
plt.legend()
plt.show()



plt.bar(list1,list2, color ='orange', width = 0.4)
plt.title("Percentage of variance explained by top two PCs with centring")
plt.show()
