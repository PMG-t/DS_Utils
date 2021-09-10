from . import _costanti as c
from . import _semanticF as s
from collections import Counter

import matplotlib.pyplot as plt
import networkx as nx
import pandas as pd
import numpy as np
import math

def gcorr(lst, g):
    m = max(lst)
    return [math.pow((v/m),g)*m for v in lst]

def rescale(lst, NewMin, NewMax):
    OldMin = min(lst)
    OldMax = max(lst)
    return [((((OldValue - OldMin) * (NewMax - NewMin)) / (OldMax - OldMin)) + NewMin) for OldValue in lst]

def rescale_dict(diz, min, max):
    tmp = rescale([v+1 for v in diz.values()],min,max)
    return {k:tmp[i] for i,k in enumerate(diz.keys())}


def sort_dict_by_key(d):
    ks = sorted(d.keys())
    return {k:d[k] for k in ks}

def sort_dict_by_value(d, reverse=True):
    return dict(sorted(d.items(), key=lambda item: item[1], reverse=reverse))

# score = combination of topic_similarities and keywords_entropy
def score(topics_frequencies, topic_sim):
    topics = list(topics_frequencies.keys())
    values = list(topics_frequencies.values())

    if len(topics) == 0:
        return 0
    elif len(topics) == 1:
        topic_similarities = max([max(topic_sim[t].values()) for t in topic_sim]) #max existing value
    else:
        topic_similarities = np.mean([topic_sim[t1][t2] for t1 in topics for t2 in topics if t1 != t2])

    return topic_similarities * entropy(values)


def entropy(arr):
    if [arr[0]] * len(arr) == arr:
        return 1.5/np.sqrt(len(arr))
    return np.std(arr)/np.sqrt(len(arr))

# calculate distances between every topic,
# by averaging top 3 similar words for each pair of topics
# returns dict: topic_dist['crime']['vegetation']=0.22
def calc_topic_sim():
    topic_sim = {x: {y: 0 for y in c._TOPIC if x != y} for x in c._TOPIC}

    for t1 in topic_sim:
        for t2 in topic_sim[t1]:
            kw1 = c._TOPIC[t1]
            kw2 = c._TOPIC[t2]
            dists = [1-s._MODEL.wv.distance(k1, k2) for k1 in kw1 for k2 in kw2]
            topic_sim[t1][t2] = np.mean(sorted(dists, reverse=True)[:3])

    return topic_sim

def city_cohesion(rw_city, reverse_topic, topic_sim):
    text = {x[0]: x[1] for x in rw_city}
    topic_count = dict(Counter([reverse_topic[t] for t in text]))
    return score(topic_count, topic_sim)


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
