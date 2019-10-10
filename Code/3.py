
#### Author - Anup Patel (M.tech CSA - 15474)

#### Import Library
import pandas as pd
import numpy as np
import networkx as nx
import operator
from sklearn.cluster import KMeans
from matplotlib.pyplot import figure
import matplotlib.pyplot as plt
from scipy.sparse import csgraph


#### Extracting Data from csv File 

circle_data=pd.read_csv("../data/11_twoCirclesData.csv")

x=circle_data['x']
y=circle_data['y']

l=len(circle_data)
circle_data=circle_data.values

#### Adjacency Matrix

sigma_square=0.0057

adjacency_matrix=np.array([[0.]*len(circle_data) for i in range(len(circle_data))])
for i in range(l):
    for j in range(l):
        if(i==j):
            adjacency_matrix[i][j]=0
        else:
            adjacency_matrix[i][j]=np.exp(-1*(np.linalg.norm(circle_data[i]-circle_data[j]))**2/sigma_square)


#### Normalized Laplacian Matrix 

laplacian=csgraph.laplacian(adjacency_matrix, normed=True)


#### Eigen Value and Eigen Vector Computation

eigen_value,eigen_vector=np.linalg.eigh(laplacian)  # Eigen Value in ascending order

eigen_vector = eigen_vector.T

#### Clustering 

colours={}
colours[0]='b'
colours[1]='g'
#kmeans Clustering
kmeans = KMeans(n_clusters=2).fit((eigen_vector[0:2]).T)
# Building the colour vector for each data point 
#print(kmeans.labels_)
cvec = [colours[label] for label in kmeans.labels_] 
  
# Plotting the clustered scatter plot 
plt.figure(figsize =(10, 10)) 
plt.scatter(x, y, c = cvec) 
plt.savefig('plots/two_circle.png')
print("Clusters Plot Saved Successfully")
