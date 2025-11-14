#
# Template for Task 3: Kmeans Clustering
#
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

# --- Your Task --- #
# import libraries as needed 
from mpl_toolkits.mplot3d import Axes3D
# --- end of task --- #

# -------------------------------------
# load data 
# note we do not need label 
data = np.loadtxt('crimerate.csv', delimiter=',')
[n,p] = np.shape(data)
sample = data[:,0:-1]
# -------------------------------------

# --- Your Task --- #
# pick a proper number of clusters 
k = 3
# --- end of task --- #



def distance_all_centers(samples, centroids) -> list: # returns a vector of all distances
     # return a list of the distance to each center
     n = samples.shape[0] # get number of rows
     k = centroids.shape[0] # get number of centers
     distances = np.zeros((n, k)) # initialize list to zero with correct sizes
     for i in range(k):
          # find the euclidian distance from a point to a center
          # axis = 1 computes the magnitude for each row
          distances[:,i] = np.linalg.norm(samples - centroids[i], axis = 1)

     return distances

# --- Your Task --- #
# implement the Kmeans clustering algorithm 
# you need to first randomly initialize k cluster centers 
# start at random points already on the chart
rand_index = np.random.choice(n, k, replace=False) # n is range, k is amount, sample without replacement
centers = sample[rand_index, :] 
# then start a loop 
max_iter = 100
for _ in range (max_iter):

     # create each cluster
     distances = distance_all_centers(sample, centers)
     # label each point with a cluster
     labels = np.argmin(distances, axis = 1) # get index of closest cluster and save to current label


     # find mean centroid for each cluster, recompute centers
     new_centers = np.zeros_like(centers) # initialize 0 list, same shape as centers
     for cluster_index in range(k): # for each cluster get the mean value
          cluster_mean = np.mean(sample[labels == cluster_index], axis=0)
          new_centers[cluster_index] = cluster_mean

     # determine if it has converged, if so stop the loop
     if np.allclose(centers, new_centers): # np.all close checks equality under a certain tolerance
          break
     
     centers = new_centers.copy()



# when clustering is done, 
# store the clustering label in `label_cluster' 
# cluster index starts from 0 e.g., 
# label_cluster[0] = 1 means the 1st point assigned to cluster 1
# label_cluster[1] = 0 means the 2nd point assigned to cluster 0
# label_cluster[2] = 2 means the 3rd point assigned to cluster 2
label_cluster = labels.copy()
# --- end of task --- #


# the following code plot your clustering result in a 2D space
pca = PCA(n_components=2)
pca.fit(sample)
sample_pca = pca.transform(sample)
idx = []
colors = ['blue','red','green','m']
for i in range(k):
     idx = np.where(label_cluster == i)
     plt.scatter(sample_pca[idx,0],sample_pca[idx,1],color=colors[i],facecolors='none')

# ---- Plot centroids with a big X ----
centers_pca = pca.transform(centers)
plt.scatter(centers_pca[:, 0], centers_pca[:, 1],
            marker='x', s=200, color='black', linewidths=3,
            label="Centroids")

plt.title(f"K-Means Clusters (PCA 2D) | K = {k}")
plt.xlabel("PC1")
plt.ylabel("PC2")
plt.show()

# 3d plot:
pca3D = PCA(n_components=3)
sample_pca3D = pca3D.fit_transform(sample)

fig = plt.figure(figsize=(8,6))
ax = fig.add_subplot(111, projection='3d')
# Plot points by cluster
for i in range(k):
    idx = np.where(label_cluster == i)
    ax.scatter(sample_pca3D[idx, 0], sample_pca3D[idx, 1], sample_pca3D[idx, 2],
               color=colors[i], alpha=0.6, label=f"Cluster {i}")
    
ax.set_title("K-Means Clusters in 3D (PCA)")
ax.set_xlabel("PC1")
ax.set_ylabel("PC2")
ax.set_zlabel("PC3")
ax.legend()
plt.show()