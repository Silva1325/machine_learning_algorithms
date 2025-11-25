from matplotlib import pyplot as plt
import pandas as pd
from sklearn.calibration import LabelEncoder
from sklearn.cluster import KMeans
from sklearn.discriminant_analysis import StandardScaler

# Import data
ds = pd.read_csv(r"4 - Clustering\Datasources\mall_customers_data.csv")

# Indepentend variables
X = ds.iloc[:,3:].values

# Using elbow method to find the optimal number of clusters
wcss = []
for i in range(1, 11):
    kmeans = KMeans(n_clusters=i, init='k-means++', random_state=42)
    kmeans.fit(X)
    wcss.append(kmeans.inertia_)

plt.figure()
plt.plot(range(1,11), wcss)
plt.xlabel("Number of cluster")
plt.ylabel("WCSS")
plt.show()
print(wcss)

# Train the model
kmeans = KMeans(n_clusters=5, init='k-means++', random_state=42, max_iter=1000)
kmeans.fit(X)
y_kmeans = kmeans.predict(X)
print(y_kmeans)

# Visualising the clusters
plt.scatter(X[y_kmeans == 0,0],X[y_kmeans == 0,1], s = 100, c = 'red', label = 'Cluster 1')
plt.scatter(X[y_kmeans == 1,0],X[y_kmeans == 1,1], s = 100, c = 'blue', label = 'Cluster 2')
plt.scatter(X[y_kmeans == 2,0],X[y_kmeans == 2,1], s = 100, c = 'green', label = 'Cluster 3')
plt.scatter(X[y_kmeans == 3,0],X[y_kmeans == 3,1], s = 100, c = 'cyan', label = 'Cluster 4')
plt.scatter(X[y_kmeans == 4,0],X[y_kmeans == 4,1], s = 100, c = 'magenta', label = 'Cluster 5')
plt.scatter(kmeans.cluster_centers_[:,0],kmeans.cluster_centers_[:,1], s = 300, c = 'yellow', label = 'Centroids')
plt.title('Clusters of customers')
plt.xlabel("Annual Income")
plt.ylabel("Spending Score (1 - 100)")
plt.legend()
plt.show()