from . import _costanti as c
from . import _semanticF as s
from collections import Counter

import matplotlib.pyplot as plt
import networkx as nx
import pandas as pd
import numpy as np
import math
import itertools

from bokeh.io import output_notebook, show, save
from bokeh.io import output_notebook, show, save
from bokeh.models import CustomJS, Div, Row, Range1d, Circle, ColumnDataSource, MultiLine, StaticLayoutProvider, EdgesAndLinkedNodes, NodesAndLinkedEdges, LabelSet
from bokeh.plotting import figure
from bokeh.plotting import from_networkx

from bokeh.io import output_notebook, show, save
from bokeh.models import Range1d, Circle, ColumnDataSource, MultiLine
from bokeh.plotting import figure
from bokeh.plotting import from_networkx
from bokeh.palettes import Blues8, Reds8, Purples8, Oranges8, Viridis8, Spectral8
from bokeh.transform import linear_cmap

def city_cohesion(rw_city, reverse_topic):
    topic_count = dict(Counter([reverse_topic[t] for t in rw_city]))
    values = list(topic_count.values())
    return s.score(values)


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

def html_to_js(title, html, js):
    with open(html, 'r') as htmlf:
        lines = [s.strip() for s in htmlf.read().split('\n') if s.strip()!='']
    pezzotto = lines[13].replace('\'','').replace('\\',' ')
    id1 = lines[11].replace('<div class="bk-root" id="','')
    id1 = id1[:id1.index('"')]
    id2 = lines[11].replace('<div class="bk-root" id="'+id1+'" data-root-id="','')
    id2 = id2[:id2.index('"')]
    id3 = pezzotto[2:pezzotto.index('":')]
    jsText = ""
    jsText += """function load_""" + title.lower() + """() {
        Bokeh.safely(function() {
          (function(root) {
            function embed_document(root) {
            var docs_json = '"""
    jsText += pezzotto + "';\n"
    jsText += """var x = document.getElementsByClassName("main-root")[0];\n"""

    jsText += """x.setAttribute('id','<1>');\n""".replace('<1>', id1)
    jsText += """x.setAttribute('data-root-id', '<2>');\n""".replace('<2>', id2)
    jsText += """render_items = [{"docid":"<3>","root_ids":["<2>"],"roots":{"<2>":"<1>"}}];\n""".replace('<1>',id1).replace('<2>',id2).replace('<3>',id3)

    jsText += """root.Bokeh.embed.embed_items(docs_json, render_items);

            }
            if (root.Bokeh !== undefined) {
              embed_document(root);
            } else {
              var attempts = 0;
              var timer = setInterval(function(root) {
                if (root.Bokeh !== undefined) {
                  clearInterval(timer);
                  embed_document(root);
                } else {
                  attempts++;
                  if (attempts > 100) {
                    clearInterval(timer);
                    console.log("Bokeh: ERROR: Unable to run BokehJS code because BokehJS library is missing");
                  }
                }
              }, 10, root)
            }
          })(window);
        });
    };"""
    with open(js, 'w') as jsf:
        jsf.write(jsText)

def link_score1(a, b):
    return np.sum([np.min([a[t], b[t]]) for t in a if t in b])

def link_score2(a, b):
    tot = 0
    for t in a:
        if t in b:
            pos1 = (len(a) - list(a.keys()).index(t)) / len(a)
            pos2 = (len(b) - list(b.keys()).index(t)) / len(b)
            tot += (pos1 * a[t]) + (pos2 * b[t])
    return tot

def sort_dict(x):
    return {k: v for k, v in sorted(x.items(), key=lambda item: item[1], reverse=True)}

def create_topic_similarity_dict(model, topics_dict):
    return {topic:{kw: 1-model.wv.distance(topic, kw) for kw in topics_dict[topic]} for topic in topics_dict}

def link_score3(a, b, sim_dict, t):
    tot = 0
    for w in dict(itertools.islice(a.items(), 10)):
        if w in dict(itertools.islice(b.items(), 10)):
            pos1 = (len(a) - list(a.keys()).index(w)) / len(a)
            pos2 = (len(b) - list(b.keys()).index(w)) / len(b)

            if not w in sim_dict[t]:
                return False
            kw1 = sim_dict[t][w] * a[w]
            kw2 = sim_dict[t][w] * b[w]

            tot += (pos1 * kw1) + (pos2 * kw2)
    return tot

def auto_gamma(lst):
    gamma = 1
    while gamma < 8:
        wr = s.gcorr(lst,gamma)
        d = s.describe(wr, print_plot=False)
        if (d['skewness']>=12):
            return wr
        gamma += 0.25


def auto_gamma2(lst0, lst=None, direction=None, param='skewness', norm=True, exclude_0=False, g=1, step=0.25, maxg=10, ming=0.1):
    lst = lst0.copy() if lst==None else lst
    lst = s.rescale(lst,0,1) if norm else lst
    if len(direction)==2:
        d = s.describe(lst, print_plot=False, print_res=False, exclude_0=exclude_0)
        if direction[0]=='<' and d[param]>direction[1] and g>=ming and g<=maxg:
            print(d[param],'1',g)
            return auto_gamma2(lst0, lst=s.gcorr(lst0,g), direction=['<', direction[1]], param=param, norm=norm, g=g-step, step=step, maxg=maxg, ming=ming)
        elif direction[0]=='>' and d[param]<direction[1] and g>=ming and g<=maxg:
            print(d[param],'2',g)
            return auto_gamma2(lst0, lst=s.gcorr(lst0,g), direction=['>', direction[1]], param=param, norm=norm, g=g+step, step=step, maxg=maxg, ming=ming)
        else:
            return (lst,g)
    elif len(direction)==3:
        if direction[0]=='in':
            d = s.describe(lst, print_plot=False, print_res=False, exclude_0=exclude_0)
            if d[param] < direction[1] and g>=ming and g<=maxg: # sotto estremo inf -> g da alzare
                print('A')
                return auto_gamma2(lst0, lst=lst, direction=['>', direction[1]], param=param, norm=norm, g=g, step=step, maxg=maxg, ming=ming)
            elif d[param] > direction[2] and g>=ming and g<=maxg: # sopra estremo sup -> g da abbassare
                print('B')
                return auto_gamma2(lst0, lst=lst, direction=['<', direction[2]], param=param, norm=norm, g=g, step=step, maxg=maxg, ming=ming)
            else:
                return (lst,g)
    else:
        print('autogamma fail')
        return None

    gamma = 1
    while gamma < 8:
        wr = s.gcorr(lst,gamma)
        d = s.describe(wr, print_plot=False)
        if (d['skewness']>=12):
            return wr
        gamma += 0.25


def generate_topic_graph(model, t, link):
    cur_cities = c._TOT_CITIES
    cur_topics = list(c._TOPIC.keys())
    title = 'Graph_' + t + '_link_' + str(link)
    threshold = 0.1
    max_words = 50

    reverse_topic = {}
    x = [reverse_topic.update({word: topic for word in c._TOPIC[topic]}) for topic in c._TOPIC]

    rwT = s.get_related_words(model, cur_cities, c._TOT_TOPIC, threshold, max_words, True)

    topic_words = {}
    for city in rwT.keys():
        topic_words[city] = {}
        for w in rwT[city]:
            if c._REVERSE_TOPIC[w] in topic_words[city]:
                topic_words[city][c._REVERSE_TOPIC[w]].append((w, rwT[city][w]))
            else:
                topic_words[city][c._REVERSE_TOPIC[w]] = [(w, rwT[city][w])]

    # Cities connection
    done = []
    entries = []
    for c1 in topic_words:
        for c2 in topic_words:
            if c1!=c2 and ((c1,c2) not in done):
                if t in topic_words[c1] and t in topic_words[c2]:
                    tot_nw1 = sum(dict(Counter([c._REVERSE_TOPIC[t] for t in rwT[c1]])).values())
                    tot_nw2 = sum(dict(Counter([c._REVERSE_TOPIC[t] for t in rwT[c2]])).values())

                    w1 = [ws[0] for ws in topic_words[c1][t]]
                    w2 = [ws[0] for ws in topic_words[c2][t]]
                    s1 = [ws[1] for ws in topic_words[c1][t]]
                    s2 = [ws[1] for ws in topic_words[c2][t]]

                    freq = (len(w1)/tot_nw1) * (len(w2)/tot_nw2)

                    a = sort_dict({x[0]: x[1] for x in topic_words[c1][t]})
                    b = sort_dict({x[0]: x[1] for x in topic_words[c2][t]})

                    val = None
                    if link==1:
                        val = link_score1(a, b)
                    elif link==2:
                        val = link_score2(a, b)
                    elif link==3:
                        val = link_score3(a, b, create_topic_similarity_dict(model, s._TOPIC), t)

                    entries.append([c1,c2,val])

                else:
                    pass
                done.append((c1,c2))
    df_start=pd.DataFrame(entries, columns=['source', 'target', 'weight']).sort_values('weight',ascending=False)
    df=df_start.copy()

    # dimensione nodo
    tfreq = {}
    for city in cur_cities:
        topic_count = dict(Counter([c._REVERSE_TOPIC[t] for t in rwT[city]]))
        tfreq[city] = topic_count[t]/sum(list(topic_count.values())) if t in topic_count else 0

    # Colore nodo, trasparenza
    tcoh = {}
    for city in cur_cities:
        if t in topic_words[city]:
            words = topic_words[city][t]
            dists = s.flatten(s.dist_matrix([w[0] for w in words], norm=False, show=False)['matrix'])
            dist_transf = s.gcorr(s.rescale(dists,0,1),2)
            coh = ((sum(dist_transf)-len(words))/2)/((len(words)**2) - len(words)) if len(words)>1 else (math.pow(s.wdist(words[0][0], t)/2,2))
            tcoh[city] = coh
        else:
            tcoh[city] = 0

    # Prime tre parole
    topn = 3
    ttopw = {city: ', '.join([w[0] for w in topic_words[city][t][:topn] if w[1]>0.2]) if t in topic_words[city] else '//' for city in cur_cities}
    ttopw = {k:ttopw[k] if ttopw[k]!='' else '//' for k in ttopw}

    G = nx.from_pandas_edgelist(df, 'source', 'target', 'weight')

    t_color = s.rgb2hex(c._TOPIC_COLOR[t])

    ## Create a network graph object with spring layout
    ## https://networkx.github.io/documentation/networkx-1.9/reference/generated/networkx.drawinlayout.spring_layout.html
    gr = from_networkx(G, nx.spring_layout)

    ## Set node coord
    coords = read_coords(s.P._BASE_DIR+r'\DS_Utils\wcities_coords.csv', cur_cities)
    gr.layout_provider = StaticLayoutProvider(graph_layout=coords)

    ## Set node attributes
    add_node_attr(gr, 'topic_freq', tfreq)
    add_node_attr(gr, 'topic_freq_rescale', s.rescale_dict(tfreq, 5, 40))
    add_node_attr(gr, 'topic_freq_correct', s.gcorr_dict(tfreq,1.25))
    add_node_attr(gr, 'topic_freq_rescaled_gcorrect', s.rescale_dict(s.gcorr_dict(tfreq, 1.25), 5, 30))

    add_node_attr(gr, 'topic_coh', tcoh)
    add_node_attr(gr, 'topic_coh_rescale', s.rescale_dict(tcoh, 0,1))
    add_node_attr(gr, 'topic_coh_gcorrect_rescale', s.gcorr_dict(s.rescale_dict(tcoh, 0,1), 1.3))

    add_node_attr(gr, 'top_words', ttopw)

    ## Set node size and color
    gr.node_renderer.glyph = Circle(size='topic_freq_rescaled_gcorrect',
                                    fill_color=t_color, fill_alpha='topic_coh_rescale')

    ## Set edge attributes
    gr.edge_renderer.data_source.data['weight_rescaled'] = s.rescale(gr.edge_renderer.data_source.data['weight'], 0, 1)

    gr.edge_renderer.data_source.data['weight_gcorrect'] = auto_gamma(gr.edge_renderer.data_source.data['weight_rescaled'])
    gr.edge_renderer.data_source.data['weight_rescaled_gcorrect'] = s.rescale(gr.edge_renderer.data_source.data['weight_gcorrect'],0,7)

    gr.edge_renderer.data_source.data['weight_gcorrect_hover'] =[v+0.2 for v in gr.edge_renderer.data_source.data['weight_gcorrect']]
    gr.edge_renderer.data_source.data['weight_rescaled_gcorrect_hover'] =[v+2 for v in gr.edge_renderer.data_source.data['weight_rescaled_gcorrect']]

    ## Set edge opacity and width
    gr.edge_renderer.glyph = MultiLine(line_alpha='weight_rescaled',
                                       line_width='weight_rescaled_gcorrect',
                                       line_color='grey')

    ## Listeneres
    gr.node_renderer.hover_glyph = Circle(size='cohesion_rescaled', fill_color=t_color, line_width=2)
    gr.node_renderer.selection_glyph = Circle(size='cohesion_rescaled', fill_color=t_color, line_width=2)

    gr.edge_renderer.selection_glyph = MultiLine(line_alpha='weight_gcorrect',
                                                   line_width='weight_rescaled_gcorrect_hover',
                                                   line_color='black')
    gr.edge_renderer.hover_glyph = MultiLine(line_alpha='weight_gcorrect',
                                                   line_width='weight_rescaled_gcorrect_hover',
                                                   line_color='black')


    ## Didascalie
    HOVER_TOOLTIPS = [("City", "@index"),("Top words", "@top_words")]
    plot = figure(tooltips = HOVER_TOOLTIPS,
                  tools="pan,wheel_zoom,save,reset", active_scroll='wheel_zoom',
                  height=550, width=1000,
                  #x_range=Range1d(-10.1, 10.1), y_range=Range1d(-10.1, 10.1),
                  title=title)

    ## Listener policy
    gr.selection_policy = NodesAndLinkedEdges()
    gr.inspection_policy = NodesAndLinkedEdges()

    ## Labels
    labels_source = ColumnDataSource({'names': tuple(coords.keys()),
                                      'lats': tuple(coords[ck][0] for ck in coords.keys()),
                                      'lngs': tuple(coords[ck][1] for ck in coords.keys())})

    labels = LabelSet(x='lats', y='lngs', text='names', source=labels_source,
                      background_fill_color='white', text_font_size='10px', background_fill_alpha=.7)

    #Add network graph to the plot
    plot.renderers.append(labels)
    plot.renderers.append(gr)

    show(plot)

    return plot
