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

data= pd.read_csv(r"C:\Users\mudit\Documents\weather-check.csv") #reading csv file\


data_clean=data.replace('-',np.NaN)
#data_clean=data.dropna(inplace=False)

cluster=data_clean[['RespondentID','Do you typically check a daily weather report?'
,'How do you typically check the weather?','A specific website or app (please provide the answer)',
'If you had a smartwatch like the soon to be released Apple Watch how likely or unlikely would you be to check the weather on that device?',
'Age', 'What is your gender?' ,'How much total combined money did all members of your HOUSEHOLD earn last year?',
'US Region']]
#variables choosen for clustering

#changing name of the variables
cluster.rename(columns={'Do you typically check a daily weather report?': 'DYTC'}, inplace=True)
cluster.rename(columns={'How do you typically check the weather?':'HDYT'},inplace=True)
cluster.rename(columns={'A specific website or app (please provide the answer)':'WORP'},inplace=True)
cluster.rename(columns={'If you had a smartwatch like the soon to be released Apple Watch how likely or unlikely would you be to check the weather on that device?':'LUW'},inplace=True)
cluster.rename(columns={'What is your gender?':'gender'},inplace=True)
cluster.rename(columns={'How much total combined money did all members of your HOUSEHOLD earn last year?':'HI'},inplace=True)
cluster.rename(columns={'US Region':'USregion'},inplace=True)


cluster.dropna(subset=['HDYT','LUW','Age','gender','HI'
,'USregion'],how='any',inplace="True")
#removing Nil Values
    
del cluster['WORP']
del cluster['RespondentID']
#dripping variables not required

number =LabelEncoder()
clustervar=cluster.copy()
#clustervar['RespondentID']=number.fit_transform(clustervar['RespondentID'].astype('str'))
#clustervar['RespondentID']=preprocessing.scale(clustervar['RespondentID'].astype('float64'))
clustervar['LUW']=number.fit_transform(clustervar['LUW'].astype('str'))
clustervar['LUW']=preprocessing.scale(clustervar['LUW'].astype('float64'))
clustervar['Age']=number.fit_transform(clustervar['Age'].astype('str'))
clustervar['Age']=preprocessing.scale(clustervar['Age'].astype('float64'))
clustervar['gender']=number.fit_transform(clustervar['gender'].astype('str'))
clustervar['gender']=preprocessing.scale(clustervar['gender'].astype('float64'))
clustervar['HI']=number.fit_transform(clustervar['HI'].astype('str'))
clustervar['HI']=preprocessing.scale(clustervar['HI'].astype('float64'))
clustervar['USregion']=number.fit_transform(clustervar['USregion'].astype('str'))
clustervar['USregion']=preprocessing.scale(clustervar['USregion'].astype('float64'))
clustervar['DYTC']=number.fit_transform(clustervar['DYTC'].astype('str'))
clustervar['DYTC']=preprocessing.scale(clustervar['DYTC'].astype('float64'))
clustervar['HDYT']=number.fit_transform(clustervar['HDYT'].astype('str'))
clustervar['HDYT']=preprocessing.scale(clustervar['HDYT'].astype('float64'))

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

#now k means for 5 clusters
model3=KMeans(n_clusters=5)
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
plt.title('Scatterplot of canonical variable for 5 clusters')
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

#END multiple steps to merge cluster assignment with clustering variables to examine
#cluster variable means by cluster


# calculate test data clustering variable means by cluster
clustergrpval = merged_test.groupby('cluster').mean()
print ("Test data clustering variable means by cluster")
print(clustergrpval)
