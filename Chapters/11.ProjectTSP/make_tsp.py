"""
Created on Wednesday 2 May 2018
Last update: Thursday 4 May 2018

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

im_name = 'Data/totoro.jpg'
n_points = 1000
metric = 'euclidean'

np.random.seed(12)

# read image
image = rgb2grey(imread(im_name)) < 0.6
n, m = image.shape

# sample points
sample_mask = image / image.sum()*n_points > np.random.rand(n, m)
coordinates = [(j, n-i) for i in range(n) for j in range(m) if sample_mask[i,j]]
distances = pairwise_distances(coordinates, metric=metric)

# save
np.save('Data/coordinates.npy', coordinates)

# make plot
fig, ax = plt.subplots()
ax.scatter(*zip(*coordinates), color=blue, s=5)
ax.set_aspect('equal')
ax.set_xticks([])
ax.set_yticks([])
fig.savefig('Figures/cities.png')
