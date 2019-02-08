# -*- coding: utf-8 -*-
"""
Created on Sun Apr 23 2017
Last update on Mon Apr 24 2017

@author: Michiel Stock

Parsing the Ghent park network
"""

import json
import geopandas as gpd
import shapely
import numpy as np

if __name__=='__main__':
    # read roads in Ghent
    streets = gpd.read_file('Data/ex_SXXm38nTMVKwsPsrBjWKF5Q6ch5zi_osm_line.geojson')
    n_streets = len(streets)
    length_streets = streets.length

    # get all parks
    polygons = gpd.read_file('Data/ex_SXXm38nTMVKwsPsrBjWKF5Q6ch5zi_osm_polygon.geojson')
    parks = polygons.loc[polygons.leisure=='park']
    parks = parks.loc[parks.geometry.area > 1e-5]  # only parks larger than 100 m^2

    # make graph
    edges = set([])
    vertices_park = set([])
    vertices = set([])

    for i in range(n_streets):
        x, y = streets.iloc[i].geometry.coords.xy
        x0, xe = x[0], x[-1]
        y0, ye = y[0], y[-1]
        v0, v1 = (x0, y0), (xe, ye)
        length = float(length_streets.iloc[i])
        edges.add((length, v0, v1))
        vertices.add(v0)
        vertices.add(v1)
        if np.any(parks.contains(shapely.geometry.asPoint(v0))):
            vertices_park.add(v0)
        if np.any(parks.contains(shapely.geometry.asPoint(v1))):
            vertices_park.add(v1)

    # put in dictionaries
    coordinates = {str(i) : e for i, e in enumerate(vertices)}
    coor_to_ind = {v : k for k, v in coordinates.items()}
    ghent_graph = {}

    for w, c0, c1 in edges:
        v0 = coor_to_ind[c0]
        v1 = coor_to_ind[c1]
        if v0 not in ghent_graph:
            ghent_graph[v0] = set([(w, v1)])
        else:
            ghent_graph[v0].add((w, v1))
        if v1 not in ghent_graph:
            ghent_graph[v1] = set([(w, v0)])
        else:
            ghent_graph[v1].add((w, v0))

    # dump in json file

    ghent_data = {
        'adjacency_list' : {k : list(v) for k, v in ghent_graph.items()},
        'coordinates' : coordinates,
        'park vertices' : [coor_to_ind[c] for c in vertices_park]
    }

    json.dump(obj=ghent_data, fp=open('Data/graph_parks_ghent.json', 'w'))
