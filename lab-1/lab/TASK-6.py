import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('cars.csv',sep=',')

X = dataset.iloc[:, :-1]
x = pd.DataFrame(X)
print(x)
x.info()
'''Y = x.convert_objects(convert_numeric=True)'''

x.mpg = pd.to_numeric(x.mpg, errors='coerce').fillna(0).astype(np.int64)
x[' cylinders'] = pd.to_numeric(x[' cylinders'], errors='coerce').fillna(0).astype(np.int64)
x[' cubicinches'] = pd.to_numeric(x[' cubicinches'], errors='coerce').fillna(0).astype(np.int64)
x[' hp'] = pd.to_numeric(x[' hp'], errors='coerce').fillna(0).astype(np.int64)
x[' weightlbs'] = pd.to_numeric(x[' weightlbs'], errors='coerce').fillna(0).astype(np.int64)
x.info()

x.columns = ['mpg', ' cylinders', ' cubicinches', ' hp', ' weightlbs', ' time-to-60', 'year']

# Eliminating null values
for i in x.columns:
    x[i] = x[i].fillna(int(x[i].mean()))
for i in x.columns:
    print(x[i].isnull().sum())


# Using the elbow method to find  the optimal number of clusters
from sklearn.cluster import KMeans

wcss = []
for i in range(1, 11):
    kmeans = KMeans(n_clusters=i, init='k-means++', max_iter=300, n_init=10, random_state=0)
    kmeans.fit(x)
    wcss.append(kmeans.inertia_)
plt.plot(range(1, 11), wcss)
plt.title('The Elbow Method')
plt.xlabel('Number of clusters')
plt.ylabel('WCSS')
plt.show()

# Applying k-means to the cars dataset
kmeans = KMeans(n_clusters=3, init='k-means++', max_iter=300, n_init=10, random_state=0)
y_kmeans = kmeans.fit_predict(x)

X = x.as_matrix(columns=None)

# Visualising the clusters
plt.scatter(X[y_kmeans == 0, 0], X[y_kmeans == 0, 1], s=100, c='red', label='US')
plt.scatter(X[y_kmeans == 1, 0], X[y_kmeans == 1, 1], s=100, c='blue', label='Japan')
plt.scatter(X[y_kmeans == 2, 0], X[y_kmeans == 2, 1], s=100, c='green', label='Europe')
plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], s=300, c='yellow', label='Centroids')
plt.title('Clusters of car brands')
plt.legend()
plt.show()