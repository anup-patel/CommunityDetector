
#### Author - Anup Patel (M.tech CSA - 15474)

import pandas as pd
import numpy as np
import networkx as nx
import operator
from sklearn.cluster import KMeans
from matplotlib.pyplot import figure


#### Extracting Data from gml File 


dolphins_data=nx.read_gml("../data/dolphins/dolphins.gml")



#### Nodes

nodes=np.array(dolphins_data.node)

### Normalized Laplacian Matrix

normalized_laplacian=nx.normalized_laplacian_matrix(dolphins_data,nodelist=None, weight='weight')
normalized_laplacian=np.array(normalized_laplacian.todense())
#print(normalized_laplacian)


#### Eigen Value and eigen Vector 
eigen_value,eigen_vector=np.linalg.eigh(normalized_laplacian)
eigen_vector = eigen_vector.T

#### Eigen Value Sorting
dict={}
sorted_dict={}
for i in range(len(eigen_value)):
    dict[eigen_value[i]]=eigen_vector[i]

sorted_dict = sorted(dict.items(), key=operator.itemgetter(0))


#### Fiedler Vector
fiedler_vector=sorted_dict[1][1]

#### Clustering
positive_cluster=[]
negative_cluster=[]
X_data = np.array(fiedler_vector).reshape((len(fiedler_vector), 1))

#kmeans Clustering
kmeans = KMeans(n_clusters=2).fit(X_data)

#Positive Cluster
for i in range(len(kmeans.labels_)):
    if (kmeans.labels_[i]==1):
        positive_cluster.append(i)

#Negative Cluster
for i in range(len(kmeans.labels_)):
    if (kmeans.labels_[i]==0):
        negative_cluster.append(i)

positive_cluster_dolphins = [nodes[i] for i in positive_cluster]
negative_cluster_dolphins = [nodes[i] for i in negative_cluster]


#### Visualisation 

import matplotlib.pyplot as plt
pos = nx.spring_layout(dolphins_data) #A dictionary with nodes as keys and positions as values.
label_dict = {}
for i in range(len(nodes)):
    label_dict[nodes[i]] = nodes[i]

nx.draw_networkx_nodes(dolphins_data, pos, positive_cluster_dolphins, node_color='r', node_shape='o', node_size=200)
nx.draw_networkx_nodes(dolphins_data, pos, negative_cluster_dolphins, node_color='g', node_shape='o', node_size=200)
nx.draw_networkx_edges(dolphins_data, pos)
nx.draw_networkx_labels(dolphins_data, pos, labels=label_dict, font_size=3)
plt.axis('off')
plt.savefig('plots/fiedler.png')
print("Visualisation Saved Successfully")

print("Cluster 1 : " + repr(len(positive_cluster_dolphins)) + " dolphins")
print(positive_cluster_dolphins)

print("Cluster 2 : " + repr(len(negative_cluster_dolphins)) + " dolphins")
print(negative_cluster_dolphins)
