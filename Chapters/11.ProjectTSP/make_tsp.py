"""
Created on Wednesday 2 May 2018
Last update: -

@author: Michiel Stock
michielfmstock@gmail.com

Create an instance of the TSP using an image.
"""

from skimage.io import imread
from skimage.color import rgb2grey
import numpy as np
from sklearn.metrics import pairwise_distances
import matplotlib.pyplot as plt

blue = '#264653'
green = '#2a9d8f'
yellow = '#e9c46a'
orange = '#f4a261'
red = '#e76f51'
black = '#50514F'

im_name = 'Data/cells.png'
n_points = 500

np.random.seed(10)

# read image
image = rgb2grey(imread(im_name)) > 0.1
n, m = image.shape

# sample points
sample_mask = image / image.sum()*n_points > np.random.rand(n, m)
coordinates = [(i, j) for i in range(n) for j in range(m) if sample_mask[i,j]]
distances = pairwise_distances(coordinates, metric='manhattan').astype(int)

# save
np.save('Data/coordinates.npy', coordinates)
np.save('Data/distances.npy', distances)

# make plot
fig, ax = plt.subplots()
ax.scatter(*zip(*coordinates), color=blue, s=5)
fig.savefig('Figures/cities.png')
