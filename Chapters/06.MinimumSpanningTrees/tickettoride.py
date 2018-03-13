"""
Created on Tuesday 13 March 2018
Last update: -

@author: Michiel Stock
michielfmstock@gmail.com

Ticket to ride data set as weighted edges
"""

edges = [(1, 'Vancouver', 'Seattle'),
          (1, 'Seattle', 'Portland'),
          (3, 'Vancouver', 'Calcary'),
          (6, 'Calcary', 'Winnipec'),
          (6, 'Winnipec', 'Sault St. Marie'),
          (4, 'Winnipec', 'Helena'),
          (4, 'Calcary', 'Helena'),
          (6, 'Seattle', 'Helena'),
          (4, 'Seattle', 'Calcary'),
          (6, 'Portland', 'Salt Lake City'),
          (5, 'Portland', 'San Francisco'),
          (5, 'San Francisco', 'Salt Lake City'),
          (3, 'San Francisco', 'Los Angeles'),
          (2, 'Los Angeles', 'Las Vegas'),
          (3, 'Los Angeles', 'Phoenix'),
          (6, 'Los Angeles', 'El Paso'),
          (3, 'Phoenix', 'El Paso'),
          (3, 'Phoenix', 'Santa Fe'),
          (3, 'Las Vegas', ' Salt Lake City'),
          (5, 'Phoenix', 'Denver'),
          (3, 'Salt Lake City', 'Denver'),
          (3, 'Helena', 'Salt Lake City'),
          (4, 'Helena', 'Duluth'),
          (4, 'Winnipec', 'Duluth'),
          (4, 'Helena', 'Denver'),
          (5, 'Helena', 'Omaha'),
          (4, 'Denver', 'Omaha'),
          (4, 'Denver', 'Kansas City'),
          (2, 'Denver', 'Santa Fe'),
          (2, 'Santa Fe', 'El Paso'),
          (3, 'Santa Fe', 'Oklahoma City'),
          (4, 'Denver', 'Oklahoma City'),
          (6, 'El Paso', 'Houston'),
          (4, 'El Paso', 'Dallas'),
          (6, 'El Paso', 'Oklahoma City'),
          (1, 'Dallas', 'Houston'),
          (2, 'Dallas', 'Oklahoma City'),
          (2, 'Kansas City', 'Oklahoma City'),
          (1, 'Omaha', 'Kansas City'),
          (2, 'Omaha', 'Duluth'),
          (3, 'Duluth', 'Chicago'),
          (4, 'Omaha', 'Chicago'),
          (6, 'Duluth', 'Toronto'),
          (3, 'Duluth', 'Sault St. Marie'),
          (5, 'Sault St. Marie', 'Montreal'),
          (2, 'Montreal', 'Boston'),
          (2, 'Boston', 'New York'),
          (3, 'Montreal', 'New York'),
          (3, 'Montreal', 'Toronto'),
          (4, 'Toronto', 'Chicago'),
          (3, 'Chicago', 'Pittsburg'),
          (2, 'Pittsburg', 'Toronto'),
          (2, 'Pittsburg', 'New York'),
          (2, 'Pittsburg', 'Washington'),
          (2, 'Washington', 'New York'),
          (2, 'Washington', 'Raleigh'),
          (2, 'Pittsburg', 'Raleigh'),
          (5, 'Pittsburg', 'Saint Louis'),
          (2, 'Kansas City', 'Saint Louis'),
          (2, 'Nashville', 'Saint Louis'),
          (2, 'Little Rock', 'Saint Louis'),
          (2, 'Oklahoma City', 'Little Rock'),
          (2, 'Little Rock', 'Dallas'),
          (2, 'Little Rock', 'Nashville'),
          (2, 'Houston', 'New Orleans'),
          (3, 'Little Rock', 'New Orleans'),
          (4, 'New Orleans', 'Atlanta'),
          (1, 'Atlanta', 'Nashville'),
          (4, 'Nashville', 'Pittsburg'),
          (2, 'Atlanta', 'Raleigh'),
          (3, 'Nashville', 'Raleigh'),
          (2, 'Raleigh', 'Charleston'),
          (2, 'Charleston', 'Atlanta'),
          (6, 'New Orleans', 'Miami'),
          (5, 'Atlanta', 'Miami'),
          (4, 'Charleston', 'Miami')
          ]

# add edges in other direction
edges += [(w, v2, v1) for w, v1, v2 in edges]
vertices = set([v1 for w, v1, v2 in edges])

if __name__ == '__main__':
    from minimumspanningtrees import kruskal

    edges_mst, cost = kruskal(vertices, edges)

    print('Total cost MST Noth-America: {}\n\n Edges:'.format(cost))
    for u1, u2 in edges_mst:
        print('{} => {}'.format(u1, u2))
