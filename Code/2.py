
#### Author - Anup Patel (M.tech CSA - 15474)

import pandas as pd
import numpy as np
import networkx as nx
from community import community_louvain
from matplotlib.pyplot import figure


#### Extracting Data from gml File 

dolphins_data=nx.read_gml("../data/dolphins/dolphins.gml")

#### Nodes

nodes=np.array(dolphins_data.node)


#### Louvain Algorithm 

import copy  #To copy dictionary

partition = community_louvain.best_partition(dolphins_data) #Best Partition
#print(partition)

#### To Check number of Community in Partition
val=[]
for key,value in partition.items():
    if(value not in val):
        val.append(value)
number_of_community=len(val) #Number of Community 
#print(number_of_community)


tmp={} #Dictionary

while(number_of_community!=2): #Run till Number of Community is not equal to 2
    
    community_list=[]
    for key,value in partition.items():
        if(value not in community_list):
            community_list.append(value)
            
    number_of_community=len(community_list)
    
    mat_size=np.max(community_list)+1
    mod=[[0.0]*mat_size for i in range(mat_size)]
    
    for i in community_list:   
        tmp=copy.deepcopy(partition)
        #print(tmp)
        for j in community_list: 
            #print(tmp)
            if(i!=j):
                for key,values in partition.items():
                    if(values==i):
                        tmp[key]=j
                mod[i][j]=community_louvain.modularity(tmp,dolphins_data)
    
    
    mod=np.array(mod) #Convert to array
    max=0 #Initial Value to find Max modularity
    max_index=(0,0) #to find index of max modularity
    
    for i in range(len(mod)):
        for j in range(len(mod[0])):
            if(mod[i][j]>max):
                max=mod[i][j]
                max_index=(i,j)
                
    #Merge Step
    tmp=copy.deepcopy(partition)
    #print(max_index)
    for key,values in tmp.items():
            if(values==max_index[0]):
                #print(j)
                partition[key]=max_index[1]
                
    number_of_community=number_of_community-1
    #print(partition)

#### Final Cluster 
final_cluster=[]
for key,value in partition.items():
    if(value not in final_cluster):
        final_cluster.append(value)

#### Key for Clusters
cluster_1=[]
cluster_2=[]
for key,value in partition.items():
    if(value==final_cluster[0]):
        cluster_1.append(key)
    else:
        cluster_2.append(key)


#### Visualization 

import matplotlib.pyplot as plt
pos = nx.spring_layout(dolphins_data) #A dictionary with nodes as keys and positions as values.
label_dict = {}
for i in range(len(dolphins_data)):
    label_dict[nodes[i]] = nodes[i]

nx.draw_networkx_nodes(dolphins_data, pos, cluster_1, node_color='r', node_shape='o', node_size=200)
nx.draw_networkx_nodes(dolphins_data, pos, cluster_2, node_color='g', node_shape='o', node_size=200)
#nx.draw(dolphins_data, with_labels=True, node_color=node_colors.ravel())
nx.draw_networkx_edges(dolphins_data, pos,alpha=0.8)
nx.draw_networkx_labels(dolphins_data, pos, labels=label_dict, font_size=3)
plt.axis('off')
#plt.show()
plt.savefig('plots/louvain.png')
print("Visualisation Saved Successfully")

print("Cluster 1:")
print(cluster_1)

print("Cluster 2:")
print(cluster_1)