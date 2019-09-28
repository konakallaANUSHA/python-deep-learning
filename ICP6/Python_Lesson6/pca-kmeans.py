import numpy as np
from sklearn.model_selection import train_test_split
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from sklearn import preprocessing
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler

# You can add the parameter data_home to wherever to where you want to download your data
dataset = pd.read_csv('CC.csv')

x = dataset.iloc[:, [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17]] # Normalization for 17 rows using index by row function

print(x.isna().sum())
x.fillna(x.mean(), inplace =True)
print(x.isna().sum())


scaler = StandardScaler()

# Fit on training set only.
scaler.fit(x)

# Apply transform to both the training set and the test set.
x_scaler = scaler.transform(x)
pca = PCA(2)
x_pca = pca.fit_transform(x_scaler)
df2 = pd.DataFrame(data=x_pca)
finaldf = pd.concat([df2],axis=1)
print(finaldf)

wcss = []
# ##elbow method to know the number of clusters
for i in range(2,7):
    kmeans = KMeans(n_clusters=i,init='k-means++',max_iter=300,n_init=10,random_state=0)
    kmeans.fit(finaldf)
   # print(kmeans.inertia_,'-------------------')
    wcss.append(kmeans.inertia_)
    score = silhouette_score(finaldf, kmeans.labels_, metric='euclidean')
    print("For n_clusters = {}, silhouette score is {})".format(i, score))
plt.plot(range(2, 7), wcss)
plt.title('the elbow method')
plt.xlabel('Number of Clusters')
plt.ylabel('Wcss')
plt.show()

nclusters = 4 # this is the k in kmeans
km = KMeans(n_clusters=nclusters)
km.fit(x)

# predict the cluster for each data point
y_cluster_kmeans = km.predict(x)
print(y_cluster_kmeans)
C = km.cluster_centers_.shape
print(C)

