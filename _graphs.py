
from . import _costanti as c

import matplotlib.pyplot as plt
import networkx as nx
import pandas as pd
import numpy as np



def flatten(t):
    return [item for sublist in t for item in sublist]


def get_common_words(rw, _cities, max_words):
    matrix = common_words_matrix(rw, _cities)
    
    edges = []
    for i in range(max_words+1):
        edges.append([])
    edges

    for i in range(len(_cities)):
        for j in range(len(_cities)):
            n = matrix[i][j]
            edges[int(n)].append((_cities[i], _cities[j]))
    
    return edges


def common_words_matrix(rw, _cities):
    dd = common_words_count(rw)
    
    matrix = np.zeros((len(_cities),len(_cities)))

    for i in range(len(_cities)):
        for j in range(len(_cities)):
            if (_cities[i], _cities[j]) in dd:
                matrix[i][j] = dd[(_cities[i], _cities[j])]

    matrix = np.triu(matrix, 1) #diagonale superiore

    return matrix


def plot_matrix(matrix):
    plt.matshow(matrix)
    plt.colorbar()
    plt.show()


def common_words_count(rw):
    dd = {}
    
    for city in rw:
        for word in rw[city]:
            
            for city2 in rw:
                if word in rw[city2]:
                    
                    arco = (city, city2)
                    if arco not in dd:
                        dd[arco] = 0
                    dd[arco] += 1
    
    return dd


def read_coords(filename, _cities):
    coords = pd.read_csv(filename)
    coords = coords[coords['name'].isin(_cities)]
    coords = coords.set_index('name')

    # sistema coordinate (scambio lat e lng)
    coords['tmp'] = coords['lat']
    coords['lat'] = coords['lng']
    coords['lng'] = coords['tmp']
    
    coords = coords[['lat', 'lng']]

    pos = coords.to_dict('index')
    pos = {c: (pos[c]['lat'], pos[c]['lng']) for c in pos}
    return pos


def plot_graph(coords, edges, fig_size):
    plt.figure(figsize=fig_size)

    G = nx.Graph()
    G.add_edges_from(flatten(edges[1:]))

    nx.draw_networkx(G, pos=coords, edgelist = [], node_size=0)
    nx.draw_networkx(G, pos=coords, nodelist = [], edgelist = edges[2], edge_color = 'gold', width=0.25)
    nx.draw_networkx(G, pos=coords, nodelist = [], edgelist = edges[3], edge_color = 'orange', width=1.25)
    nx.draw_networkx(G, pos=coords, nodelist = [], edgelist = edges[4], edge_color = 'tomato', width=3)
    nx.draw_networkx(G, pos=coords, nodelist = [], edgelist = edges[5], edge_color = 'firebrick', width=4)

    plt.show()
    