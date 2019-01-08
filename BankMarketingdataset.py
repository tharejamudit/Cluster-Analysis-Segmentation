from pandas import Series,DataFrame
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cross_validation import train_test_split
from sklearn.cluster import KMeans
from sklearn import preprocessing
import os

from sklearn import datasets
import sklearn.metrics as sm

from sklearn.preprocessing import LabelEncoder

#calling in libraries

data= pd.read_csv(r"C:\Users\mudit\Google Drive\docs\bank\Book1.csv") #reading csv file\
#dataset https://archive.ics.uci.edu/ml/datasets/Bank+Marketing

data_clean=data.dropna()
#delete observations with missing data


cluster=data_clean[['age','job','marital','education','default',
'balance', 'housing' ,'loan','contact',
'day', 'month',  'duration',  'campaign',
'pdays',  'previous', 'poutcome','y','p']]
#variables choosen for clustering

#cluster.describe()


number =LabelEncoder()
clustervar=cluster.copy()
clustervar['job']=number.fit_transform(clustervar['job'].astype('str'))
clustervar['job']=preprocessing.scale(clustervar['job'].astype('float64'))
clustervar['marital']=number.fit_transform(clustervar['marital'].astype('str'))
clustervar['marital']=preprocessing.scale(clustervar['marital'].astype('float64'))
clustervar['education']=number.fit_transform(clustervar['education'].astype('str'))
clustervar['education']=preprocessing.scale(clustervar['education'].astype('float64'))
clustervar['housing']=number.fit_transform(clustervar['housing'].astype('str'))
clustervar['housing']=preprocessing.scale(clustervar['housing'].astype('float64'))
clustervar['loan']=number.fit_transform(clustervar['loan'].astype('str'))
clustervar['loan']=preprocessing.scale(clustervar['loan'].astype('float64'))
clustervar['default']=number.fit_transform(clustervar['default'].astype('str'))
clustervar['default']=preprocessing.scale(clustervar['default'].astype('float64'))
clustervar['contact']=number.fit_transform(clustervar['contact'].astype('str'))
clustervar['contact']=preprocessing.scale(clustervar['contact'].astype('float64'))
clustervar['poutcome']=number.fit_transform(clustervar['poutcome'].astype('str'))
clustervar['poutcome']=preprocessing.scale(clustervar['poutcome'].astype('float64'))
clustervar['y']=number.fit_transform(clustervar['y'].astype('str'))
clustervar['y']=preprocessing.scale(clustervar['y'].astype('float64'))
clustervar['month']=number.fit_transform(clustervar['month'].astype('str'))
clustervar['month']=preprocessing.scale(clustervar['month'].astype('float64'))
clustervar['p']=number.fit_transform(clustervar['p'].astype('str'))
clustervar['p']=preprocessing.scale(clustervar['p'].astype('float64'))

#standardize clustering variables to have mean=0 and sd(standard deviation)=1

#split data into train and test sets
clus_train, clus_test = train_test_split(clustervar,test_size=.3,random_state=123)

# k-means cluster analysis for 1-9 clusters
from scipy.spatial.distance import cdist
#clusters=np.array(range(1,10))
clusters=range(1,10)
meandist=[]

for k in clusters:
    model=KMeans(n_clusters=k)
    model.fit(clus_train)
    clusassign=model.predict(clus_train)
    meandist.append(sum(np.min(cdist(clus_train,model.cluster_centers_,'euclidean'),axis=1))
    / clus_train.shape[0])

#plotting average distaance from observations from the cluster 
#centroid to use the elbow method to identify the number of clusters to choose 


plt.plot(clusters,meandist)
plt.xlabel("number of clusters")
plt.ylabel("average distance")
plt.title("selecting k with elbow method")
plt.show()

#now k means for 3 clusters
model3=KMeans(n_clusters=3)
model3.fit(clus_train)
clusassign=model3.predict(clus_train)
#plot of clusters
model3.labels_


from sklearn.decomposition import PCA
pca_2=PCA(2)
plot_columns=pca_2.fit_transform(clus_train)
plt.scatter(x=plot_columns[:,0], y=plot_columns[:,1], c=model3.labels_,)
plt.xlabel('canonical variable 1')
plt.ylabel('canonical variable 2')
plt.title('Scatterplot of canonical variable for 3 clusters')
plt.show()

#BEGIN multiple steps to merge cluster assignment with clustering variables to examine
#cluster variable means by cluster

#examining cluster variables

# create a unique identifier variable from the index for the 
# cluster training data to merge with the cluster assignment variable
clus_train.reset_index(level=0, inplace=True)
# create a list that has the new index variable
cluslist=list(clus_train['index'])
# create a list of cluster assignments
labels=list(model3.labels_)

# combine index variable list with cluster assignment list into a dictionary
newlist=dict(zip(cluslist, labels))
newlist

# convert newlist dictionary to a dataframe
newclus=DataFrame.from_dict(newlist,orient='index')
newclus

# rename the cluster assignment column
newclus.columns=['cluster']

# now do the same for the cluster assignment variable
# create a unique identifier variable from the index for the 
# cluster assignment dataframe 
# to merge with cluster training data

newclus.reset_index(level=0,inplace=True)

# merge the cluster assignment dataframe with the cluster training variable dataframe
# by the index variable
merged_train=pd.merge(clus_train,newclus,on='index')
merged_train.head(n=100)

# cluster frequencies
merged_train.cluster.value_counts()

#END multiple steps to merge cluster assignment with clustering variables to examine
#cluster variable means by cluster


#calculating clustering variable means by cluster
clustergrp=merged_train.groupby('cluster').mean()
print('clustering variable means by cluster')
print(clustergrp)


#now validation

#VALIDATION
#BEGIN multiple steps to merge cluster assignment with clustering variables to examine
#cluster variable means by cluster in test data set

# create a variable out of the index for the cluster training dataframe to merge on
clus_test.reset_index(level=0, inplace=True)
# create a list that has the new index variable
cluslistval=list(clus_test['index'])
# create a list of cluster assignments
#labelsval=list(clusassignval)
labelsval=list(cluslistval)
# combine index variable list with labels list into a dictionary
#newlistval=dict(zip(cluslistval, clusassignval))
newlistval=dict(zip(cluslistval, labelsval))
newlistval
# convert newlist dictionary to a dataframe
newclusval=DataFrame.from_dict(newlistval, orient='index')
newclusval
# rename the cluster assignment column
newclusval.columns = ['cluster']
# create a variable out of the index for the cluster assignment dataframe to merge on
newclusval.reset_index(level=0, inplace=True)
# merge the cluster assignment dataframe with the cluster training variable dataframe
# by the index variable
merged_test=pd.merge(clus_test, newclusval, on='index')
# cluster frequencies
merged_test.cluster.value_counts()
"""
END multiple steps to merge cluster assignment with clustering variables to examine
cluster variable means by cluster
"""

# calculate test data clustering variable means by cluster
clustergrpval = merged_test.groupby('cluster').mean()
print ("Test data clustering variable means by cluster")
print(clustergrpval)




