# -*- coding: utf-8 -*-
"""
Created on Mon Mar 17 2017
Last update on -

@author: Michiel Stock

The graph of the game Ticket to Ride
"""

edges = [
("Atlanta", "Charleston", 2),
("Atlanta", "Miami", 5),
("Atlanta", "Nashville", 1),
("Atlanta", "New Orleans", 4),
("Atlanta", "New Orleans", 4),
("Atlanta", "Raleigh", 2),
("Boston", "Montreal", 2),
("Boston", "Montreal", 2),
("Boston", "New York", 2),
("Boston", "New York", 2),
("Calgary", "Helena", 4),
("Calgary", "Seattle", 4),
("Calgary", "Vancouver", 3),
("Calgary", "Winnipeg", 6),
("Charleston", "Miami", 4),
("Charleston", "Raleigh", 1),
("Chicago", "Duluth", 3),
("Chicago", "Omaha", 4),
("Chicago", "Pittsburgh", 3),
("Chicago", "Pittsburgh", 3),
("Chicago", "Saint Louis", 2),
("Chicago", "Saint Louis", 2),
("Chicago", "Toronto", 4),
("Dallas", "El Paso", 4),
("Dallas", "Houston", 1),
("Dallas", "Houston", 1),
("Dallas", "Little Rock", 2),
("Dallas", "Oklahoma City", 2),
("Dallas", "Oklahoma City", 2),
("Denver", "Helena", 4),
("Denver", "Kansas City", 4),
("Denver", "Kansas City", 4),
("Denver", "Oklahoma City", 4),
("Denver", "Omaha", 4),
("Denver", "Phoenix", 4),
("Denver", "Salt Lake City", 3),
("Denver", "Salt Lake City", 3),
("Denver", "Santa Fe", 2),
("Duluth", "Helena", 6),
("Duluth", "Omaha", 2),
("Duluth", "Omaha", 2),
("Duluth", "Sault St. Marie", 3),
("Duluth", "Toronto", 6),
("Duluth", "Winnipeg", 4),
("El Paso", "Houston", 6),
("El Paso", "Los Angeles", 6),
("El Paso", "Oklahoma City", 5),
("El Paso", "Phoenix", 3),
("El Paso", "Santa Fe", 2),
("Helena", "Calgary", 4),
("Helena", "Omaha", 5),
("Helena", "Seattle", 6),
("Helena", "Winnipeg", 4),
("Houston", "New Orleans", 2),
("Kansas City", "Oklahoma City", 2),
("Kansas City", "Oklahoma City", 2),
("Kansas City", "Omaha", 1),
("Kansas City", "Omaha", 1),
("Kansas City", "Saint Louis", 2),
("Kansas City", "Saint Louis", 2),
("Las Vegas", "Los Angeles", 2),
("Las Vegas", "Salt Lake City", 3),
("Little Rock", "Nashville", 3),
("Little Rock", "New Orleans", 3),
("Little Rock", "Oklahoma City", 2),
("Little Rock", "Saint Louis", 2),
("Los Angeles", "Phoenix", 2),
("Los Angeles", "San Francisco", 3),
("Los Angeles", "San Francisco", 3),
("Miami", "New Orleans", 6),
("Montreal", "New York", 3),
("Montreal", "Sault St. Marie", 5),
("Montreal", "Toronto", 3),
("Nashville", "Pittsburgh", 4),
("Nashville", "Raleigh", 3),
("Nashville", "Saint Louis", 2),
("New York", "Pittsburgh", 2),
("New York", "Pittsburgh", 2),
("New York", "Washington DC", 2),
("New York", "Washington DC", 2),
("Phoenix", "Santa Fe", 3),
("Pittsburgh", "Saint Louis", 5),
("Portland", "Salt Lake City", 6),
("Portland", "San Francisco", 5),
("Portland", "San Francisco", 5),
("Portland", "Seattle", 1),
("Portland", "Seattle", 1),
("Raleigh", "Washington DC", 2),
("Salt Lake City", "San Francisco", 5),
("Salt Lake City", "San Francisco", 5),
("Sault St. Marie", "Toronto", 2),
("Sault St. Marie", "Winnipeg", 6),
("Seattle", "Vancouver", 1)]

# make edges
edges = [(w, c1, c2) for c1, c2, w in edges] + [(w, c2, c1) for c1, c2, w in edges]
edges = list(set(edges))  # remove duplicates

# make vertices
vertices = set([])
for w, c1, c2 in edges:
    vertices.add(c1)
    vertices.add(c2)
vertices = list(vertices)
vertices.sort()
