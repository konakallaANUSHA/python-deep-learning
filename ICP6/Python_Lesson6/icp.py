import numpy as np
import pandas as pd
from mpl_toolkits.mplot3d import Axes3D
from sklearn import preprocessing
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import LabelEncoder, StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns


dataset = pd.read_csv('CC.csv')


x = dataset.iloc[:, [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17]] # Normalization for 17 rows using index by row function

print(x.isna().sum())
x.fillna(x.mean(), inplace =True)
print(x.isna().sum())


scaler = preprocessing.StandardScaler()
scaler.fit(x)
X_scaled_array = scaler.transform(x)
X_scaled = pd.DataFrame(X_scaled_array, columns = x.columns)



# ##elbow method to know the number of clusters
wcss = []
for i in range(2,7):
    kmeans = KMeans(n_clusters=i,init='k-means++',max_iter=300,n_init=10,random_state=0)
    kmeans.fit(X_scaled)
   # print(kmeans.inertia_,'-------------------')
    wcss.append(kmeans.inertia_)
    score = silhouette_score(X_scaled, kmeans.labels_, metric='euclidean')
    print("For n_clusters = {}, silhouette score is {})".format(i, score))

plt.plot(range(2,7),wcss)
plt.title('the elbow method')
plt.xlabel('Number of Clusters')
plt.ylabel('Wcss')
plt.show()




nclusters = 3 # this is the k in kmeans
km = KMeans(n_clusters=nclusters)
km.fit(x)

# predict the cluster for each data point
y_cluster_kmeans = km.predict(x)
print(y_cluster_kmeans)
C = km.cluster_centers_.shape
print(C)

