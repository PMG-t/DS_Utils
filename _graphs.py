from . import _costanti as c
from . import _semanticF as s
from collections import Counter

import matplotlib.pyplot as plt
import networkx as nx
import pandas as pd
import numpy as np
import math

def city_cohesion(rw_city, reverse_topic, topic_sim):
    # text = {x[0]: x[1] for x in rw_city}
    topic_count = dict(Counter([reverse_topic[t] for t in rw_city]))
    return s.score(topic_count, topic_sim)


# def flatten(t):
#     return [item for sublist in t for item in sublist]


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


def plot_graph(coords, edges, fig_size=(20,12)):
    plt.figure(figsize=fig_size)

    G = nx.Graph()
    G.add_edges_from(flatten(edges[1:]))

    nx.draw_networkx(G, pos=coords, edgelist = [], node_size=0)
    nx.draw_networkx(G, pos=coords, nodelist = [], edgelist = edges[2], edge_color = 'gold', width=0.25)
    nx.draw_networkx(G, pos=coords, nodelist = [], edgelist = edges[3], edge_color = 'orange', width=1.25)
    nx.draw_networkx(G, pos=coords, nodelist = [], edgelist = edges[4], edge_color = 'tomato', width=3)
    nx.draw_networkx(G, pos=coords, nodelist = [], edgelist = edges[5], edge_color = 'firebrick', width=4)

    plt.show()


def add_node_attr(gr, attname, diz):
    gr.node_renderer.data_source.data[attname] = [diz[idx] for idx in list(gr.node_renderer.data_source.data['index'])]

def add_edge_att(gr, attname, triple):
    edgedf = pd.DataFrame([[gr.edge_renderer.data_source.data['start'][i], gr.edge_renderer.data_source.data['end'][i]] for i in range(len(gr.edge_renderer.data_source.data['start']))])
    ### ...
