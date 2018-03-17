"""
Created on Thursday 15 March 2018
Last update: Saturday 17 March 2018

@author: Michiel Stock
michielfmstock@gmail.com

Image segmentation using minmum spanning trees
"""

from skimage.color import rgb2grey
from skimage import io
from skimage.transform import rescale
from minimumspanningtrees import kruskal, USF
import numpy as np
import matplotlib.pyplot as plt
from skimage.morphology import reconstruction
from scipy.ndimage import gaussian_filter

# load an example network
image = io.imread('Figures/otoliths.jpg', as_grey=True)
image = rescale(image, 0.4)

# process the image
image = gaussian_filter(image, sigma=4)
seed = np.copy(image)
seed[1:-1, 1:-1] = image.max()
mask = image
image = reconstruction(seed, mask, method='erosion')

# make graph of the image where the pixels are the vertices which are connected
# if the are neighbors on the images
edges = set([])

n, m = image.shape
vertices = set([(i, j) for i in range(n) for j in range(m)])

for i in range(n-1):
    for j in range(m-1):
        cost_right = abs(image[i,j] - image[i,j+1])
        edges.add((cost_right, (i, j), (i, j+1)))
        edges.add((cost_right, (i, j+1), (i, j)))
        cost_down = abs(image[i,j] - image[i+1,j])
        edges.add((cost_down, (i, j), (i+1, j)))
        edges.add((cost_down, (i+1, j), (i, j)))

# note that edges are already sorted
mst_edges, mst_cost = kruskal(vertices, edges, add_weights=True)

cutoff = 0.0125
usf = USF(vertices)

for cost, v1, v2 in mst_edges:
    if cost < cutoff:
        usf.union(v1, v2)


image_segmented = np.zeros_like(image)
labels_map = {}

for i in range(n):
    for j in range(m):
        root = usf.find((i, j))
        if root not in labels_map:  # add root
            labels_map[root] = len(labels_map)
        image_segmented[i, j] = labels_map[root]

plt.imshow(image_segmented, cmap='jet')
plt.savefig('Figures/otoliths_segmented.png')
