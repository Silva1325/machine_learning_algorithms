
from matplotlib import pyplot as plt
import pandas as pd
from sklearn.cluster import AgglomerativeClustering
import scipy.cluster.hierarchy as sch

# Import data
ds = pd.read_csv(r"4 - Clustering\Datasources\mall_customers_data.csv")

# Indepentend variables
X = ds.iloc[:,3:].values

# using the dendogram to find the optimal number of clusters
dendogram = sch.dendrogram(sch.linkage(X,method='ward'))
plt.title('Clusters of customers')
plt.xlabel("Customers")
plt.ylabel("Euclidean Distance")
plt.show()

hc = AgglomerativeClustering(n_clusters=5, linkage='ward').fit(X)
y_hc = hc.fit_predict(X)
print(y_hc)

# Visualising the clusters
plt.scatter(X[y_hc == 0,0],X[y_hc == 0,1], s = 100, c = 'red', label = 'Cluster 1')
plt.scatter(X[y_hc == 1,0],X[y_hc == 1,1], s = 100, c = 'blue', label = 'Cluster 2')
plt.scatter(X[y_hc == 2,0],X[y_hc == 2,1], s = 100, c = 'green', label = 'Cluster 3')
plt.scatter(X[y_hc == 3,0],X[y_hc == 3,1], s = 100, c = 'cyan', label = 'Cluster 4')
plt.scatter(X[y_hc == 4,0],X[y_hc == 4,1], s = 100, c = 'magenta', label = 'Cluster 5')
plt.title('Clusters of customers')
plt.xlabel("Annual Income")
plt.ylabel("Spending Score (1 - 100)")
plt.legend()
plt.show()



