"""
Created on Sunday 28 January 2018
Last update: Sunday 11 March 2018

@author: Michiel Stock
michielfmstock@gmail.com

Module for transfering the color between two images
"""

from optimal_transport import compute_optimal_transport
import numpy as np
from skimage import io
from sklearn.cluster import MiniBatchKMeans as KMeans
from sklearn.preprocessing import StandardScaler
from collections import Counter
from sklearn.metrics.pairwise import pairwise_distances
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_style('white')

image_name1 = 'Figures/butterfly3.jpg'
image_name2 = 'Figures/butterfly2.jpg'

n_clusters = 400

def clip_image(im):
    """
    Clips an image such that its values are between 0 and 255
    """
    return np.maximum(0, np.minimum(im, 255))

class Image():
    """simple class to work with an image"""
    def __init__(self, image_name, n_clusters=100, use_location=True):
        super(Image, self).__init__()
        self.image = io.imread(image_name) + 0.0
        self.shape = self.image.shape
        n, m, _ = self.shape
        X = self.image.reshape(-1, 3)
        if use_location:
            col_indices = np.repeat(np.arange(n), m).reshape(-1,1)
            row_indices = np.tile(np.arange(m), n).reshape(-1,1)
        #self.standardizer = StandardScaler()
        #self.standardizer.fit_transform(
            self.X = np.concatenate([X, row_indices, col_indices], axis=1)
        else: self.X = X
        self.kmeans = KMeans(n_clusters=n_clusters)
        self.kmeans.fit(self.X)

    def compute_clusted_image(self, center_colors=None):
        """
        Returns the image with the pixels changes by their cluster center

        If center_colors is provided, uses these for the clusters, otherwise use
        centers computed by K-means.
        """
        clusters = self.kmeans.predict(self.X)
        if center_colors is None:
            X_transformed = self.kmeans.cluster_centers_[clusters,:3]
        else:
            X_transformed = center_colors[clusters,:3]
        return clip_image(X_transformed).reshape(self.shape)

    def get_color_distribution(self):
        """
        Returns the distribution of the colored pixels

        Returns:
            - counts : number of pixels in each cluster
            - centers : colors of every cluster center
        """
        clusters = self.kmeans.predict(self.X)
        count_dict = Counter(clusters)
        counts = np.array([count_dict[i] for i in range(len(count_dict))],
                        dtype=float)
        centers = self.kmeans.cluster_centers_
        return counts, clip_image(centers[:,:3])

print('loading and clustering images...')
image1 = Image(image_name1, n_clusters=n_clusters)
image2 = Image(image_name2, n_clusters=n_clusters)

r, X1 = image1.get_color_distribution()
c, X2 = image2.get_color_distribution()

C = pairwise_distances(X1, X2, metric="sqeuclidean")

print('performing optimal transport...')
P, d = compute_optimal_transport(C, r/r.sum(), c/c.sum(), 1e-2)

sns.clustermap(P, row_colors=X1/255, col_colors=X2/255,
        yticklabels=[], xticklabels=[])
plt.savefig('Figures/color_mapping.png')

print('computing and plotting color distributions...')
X1to2 = P.sum(1)**-1 * P @ X2
X2to1 = P.sum(0)**-1 * P.T @ X1

fig, axes = plt.subplots(nrows=2, ncols=2)
axes[0, 0].imshow(image1.image/255)
axes[0, 1].imshow(image2.image/255)

axes[1, 0].imshow(image1.compute_clusted_image(X1to2)/255)
axes[1, 1].imshow(image2.compute_clusted_image(X2to1)/255)

for ax in axes.flatten():
    ax.set_xticks([])
    ax.set_yticks([])

fig.savefig('Figures/color_transfer.png')
