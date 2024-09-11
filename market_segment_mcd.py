import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import LabelEncoder
from sklearn.cluster import KMeans
from scipy.cluster.hierarchy import linkage, dendrogram
import matplotlib.pyplot as plt
from statsmodels.api import Logit

# Load the McDonald's data (example assuming a CSV file)
mcdonalds = pd.read_csv("mcdonalds.csv")  # Replace with the correct data source

# Overview of the data
print(mcdonalds.columns)
print(mcdonalds.shape)
print(mcdonalds.head(3))

# Convert Yes/No columns to binary (0/1)
MD_x = mcdonalds.iloc[:, 0:11].apply(lambda x: (x == "Yes").astype(int))

# Compute the column means
col_means = MD_x.mean().round(2)
print(col_means)

# Perform PCA
pca = PCA()
MD_pca = pca.fit_transform(MD_x)

# Explained variance ratio and components
print("Explained Variance Ratios:", pca.explained_variance_ratio_)
print("Cumulative Variance:", np.cumsum(pca.explained_variance_ratio_))

# Plot PCA projection
plt.scatter(MD_pca[:, 0], MD_pca[:, 1], color='grey')
plt.xlabel("PC1")
plt.ylabel("PC2")
plt.title("PCA Projection")
plt.show()

# Perform KMeans clustering with 2 to 8 clusters
cluster_range = range(2, 9)
inertia = []
for k in cluster_range:
    kmeans = KMeans(n_clusters=k, n_init=10, random_state=1234).fit(MD_x)
    inertia.append(kmeans.inertia_)

# Plot inertia to find the optimal number of clusters (Elbow Method)
plt.plot(cluster_range, inertia, marker='o')
plt.xlabel("Number of Clusters")
plt.ylabel("Inertia")
plt.title("KMeans Clustering - Elbow Method")
plt.show()

# Assuming 4 clusters is optimal
kmeans_4 = KMeans(n_clusters=4, n_init=10, random_state=1234).fit(MD_x)
labels_4 = kmeans_4.labels_

# Plot histogram of clusters
plt.hist(labels_4, bins=np.arange(5) - 0.5, edgecolor='black')
plt.xticks(range(4))
plt.xlabel("Cluster")
plt.ylabel("Count")
plt.title("Cluster Distribution")
plt.show()

# Hierarchical clustering (optional visualization)
hclust = linkage(MD_x.T, method='ward')
dendrogram(hclust)
plt.title("Hierarchical Clustering Dendrogram")
plt.show()

# Logistic regression for segmentation (example using 'Like' as target)
mcdonalds['Like_n'] = 6 - mcdonalds['Like'].apply(lambda x: int(x))

features = mcdonalds.iloc[:, 0:11]
X = pd.get_dummies(features, drop_first=True)
y = mcdonalds['Like_n']

logit_model = Logit(y, X).fit()
print(logit_model.summary())

# Predict cluster memberships
mcdonalds['Cluster'] = labels_4

# Plot a mosaicplot (optional)
plt.scatter(mcdonalds['Age'], mcdonalds['Like_n'], c=labels_4)
plt.xlabel('Age')
plt.ylabel('Like')
plt.title('Age vs Like by Cluster')
plt.show()
