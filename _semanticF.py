# Git repository link
# https://github.com/PimpMyGit/DS_Utils

import os
import io
import re
import math
import glob
import scipy
import heapq
import pickle
import zipfile
import unidecode
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from more_itertools import split_at
from pathlib import Path
from fastcluster import linkage
from termcolor import colored
from wordcloud import WordCloud
from PIL import Image
from sklearn.cluster import DBSCAN
from gensim.models import Word2Vec
from gensim.models.phrases import Phrases, Phraser
from scipy.spatial.distance import pdist, squareform, cosine
from . import _paths as _PATHS
from . import _costanti as C
#%matplotlib inline


#------------------------------------------------------------------------------#

class Paths:

    #setup your paths
    def __init__(self):
        self._BASE_DIR = _PATHS._BASE_DIR
        self._COCA_PATH = _PATHS._COCA_PATH
        self._MODEL_PATH = _PATHS._BASE_DIR + 'w2v_mc50_bg_npmi05.pkl'
        self._LEXICON_PATH = _PATHS._BASE_DIR + '\\Lexicon\\lexicon_lemmatize.csv'


P = Paths()


# THIS MODULE NAME
_MODULE_NAME = '_semanticF'

# CORPUS CATEGORIES
_COCA_CATEGORIES = ['Academic', 'Blogs', 'Fiction', 'Magazine', 'Movies', 'Newspaper', 'Spoken']
_DICT_CATEGORIES = {
    'Academic': 'acad',
    'Blogs': 'blog',
    'Fiction': 'fic',
    'Magazine': 'mag',
    'Movies': 'tvm',
    'Newspaper': 'news',
    'Spoken': 'spok'
}
# LEXICON
def load_lexicon(drop_na=True):
    path = P._LEXICON_PATH
    if Path(path).is_file():
        lex = pd.read_csv(path, sep='\t')
        if drop_na:
            lex = lex.dropna()
        return lex
    else:
        print('Lexicon not found')
        return None


_LEXICON = load_lexicon()
# W2V MODEL
if Path(P._MODEL_PATH).is_file():
    _MODEL = pickle.load(open(P._MODEL_PATH, 'rb'))
else:
    _MODEL = None
    print('Model not found')
# HEADER GENERALI DEI DATAFRAME
_WLP_HEADER = ['original', 'standard', 'y']
# Y-TAG VARI
_STOP_TAG = 'ystop'
_LEMMA_TAG = '^n|^jj|^vv|^r'
_NEGATION_TAG = '^xx'
_SYMBOLS_TAG = ['y','ge','.'] #PEZZOTTO
# SEABORN
_SNS_FIG_SIZE=_fig_size = (10,10)
sns.set(rc={'figure.figsize':_SNS_FIG_SIZE})



#------------------------------------------------------------------------------#

_CITIES = C._CITIES
_TOPIC = C._TOPIC

#------------------------------------------------------------------------------#

class Utils:
    def __init__(self):
        pass

    def get_element_in_list(self, list, element, mode='like'):
        result = []
        for idx,el in enumerate(list):
            if mode=='equal':
                if el==element:
                    result.append(idx)
            elif mode=='like':
                if element in el:
                    result.append(el)
        return result

    def byte2str(self, bytes):
        return str(bytes)[2:-1]

    def read_inzip_files(self, archive, files='All'):
        files = archive.namelist() if files=='All' else files
        content = {}
        for f in archive.namelist():
            if f in files:
                content[f] = self.read_inzip_file(archive, f)
        return content

    def read_inzip_file(self, archive, file):
        with archive.open(file) as f:
            return f.read()

    def pickle_save(self, filename, obj, subfolder='models'):
        path = P._BASE_DIR + filename + '.pkl'
        f = open(path, 'wb')
        pickle.dump(obj, f)
        f.close()
        self.throw_msg('done', 'object saved in ' + path, 'pickle_save')


    def pickle_load(self, filename):
        path = P._BASE_DIR + filename + '.pkl'
        f = open(path + '.pkl', 'rb')
        obj = pickle.load(f)
        f.close()
        return obj

    def throw_msg(self, category, message, from_function):
        category_colour = {
            'done':'blue',
            'success':'green',
            'warn':'yellow',
            'error':'red'
        }
        if category.lower() in category_colour:
            full_message = category.upper() + ' → ' + _MODULE_NAME + ' in ' + from_function + '() → ' + message
            print(colored(full_message, category_colour[category.lower()]))
        else:
            print(self.throw_msg('error', 'message category muste be in ' + ', '.join(list(category_colour.keys())), self.throw_msg.__name__))

#------------------------------------------------------------------------------#

U = Utils()

def get_files(category, type='text'):
    archive = zipfile.ZipFile(P._COCA_PATH, 'r')
    fn = 'COCA/' + category.capitalize() + '/' + type.lower() + '_' + _DICT_CATEGORIES[category.capitalize()]
    fn = U.get_element_in_list(archive.namelist(), fn, mode='like')[0]
    print(fn)
    content = io.BytesIO(archive.read(fn))
    subarchive = zipfile.ZipFile(content)
    return U.read_inzip_files(subarchive)

def format_wlp(wlp_content):
    wlp_content = U.byte2str(wlp_content)
    wlp_content = wlp_content.replace('\\r','')
    return wlp_content

def wlp_getlines(wlp_content, do_format=True):
    if do_format:
        wlp_content = format_wlp(wlp_content)
    lines = wlp_content.split('\\n')
    return lines

def wlp_splitlines(lines):
    return [line.split('\\t')[1:] for line in lines]

def wlp_to_csv(filename, wlp_content, do_format=True, write_csv=True):
    lines = wlp_getlines(wlp_content, do_format=do_format)
    entries = wlp_splitlines(lines)
    df = pd.DataFrame(entries, columns=_WLP_HEADER)
    if write_csv:
        df.to_csv(P._BASE_DIR+filename+'.csv', sep='\t', index=False)
    return df

def massivo_csv(type='wlp'):
    tutti = []
    for category in list(_DICT_CATEGORIES.keys()):
        if category=='Blogs':
            for n in range(1,35):
                tutti.append(P._BASE_DIR + '\\' + category + '\\' + type + '_' + _DICT_CATEGORIES[category.capitalize()] + '_' + str(n).zfill(2) + '.csv')
        else:
            for year in range(1990,2020):
                tutti.append(P._BASE_DIR + '\\' + category + '\\' + type + '_' + _DICT_CATEGORIES[category.capitalize()] + '_' + str(year) + '.csv')
    return tutti


def load_csv(category, year, type='wlp', drop_na=True):
    path = P._BASE_DIR + '\\' + category + '\\' + type + '_' + _DICT_CATEGORIES[category.capitalize()] + '_' + str(year) + '.csv'
    df = pd.read_csv(path, sep='\t')
    if drop_na:
        df = df.dropna()
    return df

#ordine 1
def set_y_stop(df, standard_col='standard', y_col='y', stop_tag=_STOP_TAG, sep=['.', ';', '?', '!']):
    y_cols = [y_col] if type(y_col) is str else y_col
    for y_col in y_cols:
        df[y_col] = df.apply(lambda row: stop_tag if row[y_col] in _SYMBOLS_TAG and row[standard_col] in sep else row[y_col], axis=1)
    return df

#ordine 2
def first_y_df(wlp_df, y_col='y'):
    wlp_df['first_y'] = wlp_df['y'].apply(lambda y: first_y(str(y)))
    return wlp_df.copy()

def first_y(y):
    match = re.findall('([a-zA-Z0-9]+)', y)
    if len(match)>0:
        return match[0].lower()
    else:
        return ''

def lemmatize_df(df, lemma_col='first_y', tag=_LEMMA_TAG):
    if (lemma_col=='first_y') and (lemma_col not in list(df.columns)):
        df = first_y_df(df)
        df = set_y_stop(df, y_col=['y', 'first_y'])
    return select(df.dropna(), {lemma_col:['re', tag+'|'+_STOP_TAG]})

def del_symbols_df(df, lemma_col='first_y', tag=_SYMBOLS_TAG):
    tag = [tag] if type(tag) is str else tag
    if (lemma_col=='first_y') and (lemma_col not in list(df.columns)):
        df = first_y_df(df)
        df = set_y_stop(df, y_col=['y', 'first_y'])
    return select(df.dropna(), {lemma_col:['!']+tag})

def preprocess_df(df, preprocess='lemmatize', process_col='first_y'):
    if preprocess=='symbols':
        return del_symbols_df(df, lemma_col=process_col)
    elif preprocess=='lemmatize':
        return lemmatize_df(df, lemma_col=process_col)
    else:
        print('_semanticF: preprocess_df(): preprocess tecnique must be \'lemmatize\' or \'symbols\']')

#ordine 3
def get_sentences(dfs, sentence_col='standard', sep=['.', ';', '?', '!'], preprocess='lemmatize'):
    sentences = []
    dfs = [dfs] if type(dfs) is not list else dfs
    if preprocess:
        print('starting df preprocess ...')
        dfs = [preprocess_df(df, preprocess) for df in dfs]
    for df in dfs:
        raw_sentences = list(filter(None, list(split_at(list(df[sentence_col]), lambda x: x in sep))))
        sentences = [[word.replace('-', '_') for word in sentence] for sentence in raw_sentences]
    return sentences


def generate_bi_grams(sentences, **kwargs):
    phrases = Phrases(sentences, **kwargs)
    phrases_model = Phraser(phrases)
    return [phrases_model[sent] for sent in sentences]

def load_model():
    pass

def save_model():
    pass

def w2v_df(dfs, sentence_col='standard', sep='.',
    preprocess='lemmatize', min_count=5, vector_size=300,
    model='SG', model_filename='default_model'):
    print('getting sentences ...')
    sentences = get_sentences(dfs, sentence_col=sentence_col, sep=sep, preprocess=preprocess)
    print('done sentences!')
    print('getting bi-grams ...')
    bi_gram_sentences = generate_bi_grams(sentences,
                                        min_count=10,
                                        threshold=0.5,
                                        progress_per=1000,
                                        scoring='npmi'
                                        )
    print('done bi-grams!')
    w2v_model = w2v(bi_gram_sentences, min_count=min_count, vector_size=vector_size, model=model)
    if save_model:
        U.pickle_save(model_filename, w2v_model)
    return w2v_model

def w2v(sentences, min_count=5, vector_size=300, model='SG'):
    model = 0 if model=='CBOW' else 1
    print('building model ...')
    w2v_model = Word2Vec(sentences, min_count=min_count, vector_size=vector_size, sg=model)
    print('model completed ...')
    return w2v_model

# get list of most similar words to a given vector
def word_similarities(model, vector, topn=10):
    if isinstance(model, Word2Vec):
        model = model.wv
    # i -> index of word in model vocab
    # x -> array of word in model
    #1- => similarity
    _dict = { model.index_to_key[i]:1-cosine(model[i], vector) for i in list(range(len(model)))}
    ordered_list = {k: _dict[k] for k in heapq.nlargest(topn, _dict, key=_dict.get)}
    return ordered_list


# get the most similar words to each single entity
# - model: w2v model
# - entities: list of strings
# - total_words: list of strings
# - threshold: minimum similarity, in [0,1]
# - topn: max words for each entity (if -1, show every word)
# - include_sim: boolean, includes similarity values
def get_related_words(model, entities, total_words, threshold, topn, include_sim):
    res = {}
    for entity in entities:
        res[entity] = _get_related_words_single(model, entity, total_words, threshold, topn, include_sim)

    return res

def _get_related_words_single(model, entity, total_words, threshold, topn, include_sim):
    tmp = [1-model.wv.distance(w, entity) for w in total_words]
    tw2 = total_words.copy()
    top_words = []

    if topn==-1:
        topn = len(total_words)

    for j in range(topn):
        i = tmp.index(np.max(tmp))
        if tmp[i] > threshold:
            value = (tw2[i], tmp[i])
            if not include_sim:
                value = value[0]
            top_words.append(value)
        else:
            break
        del tmp[i]
        del tw2[i]
    return top_words


def wvec(w, model=_MODEL):
    return model.wv[w]

def wsum(w1, w2, model=_MODEL):
    w1 = model.wv[w1] if type(w1) is str else w1
    w2 = model.wv[w2] if type(w2) is str else w2
    return w1 + w1

def wdiff(w1, w2, model=_MODEL):
    w1 = model.wv[w1] if type(w1) is str else w1
    w2 = model.wv[w2] if type(w2) is str else w2
    return np.subtract(w1,w2)

def wnorm(w, model=_MODEL):
    vec = model.wv[w] if type(w) is str else w
    return math.sqrt(sum([math.pow(v,2) for v in vec]))

def wsim(pos, neg=[], topn=10, thresh=0, low_thresh=0, comp=-1, model=_MODEL, yfilter=None, fill=None):
    fill = True if (thresh==0 and fill==None) else fill
    pos = [pos] if type(pos) is str else pos
    neg = [neg] if type(neg) is str else neg
    sim0 = list(model.wv.most_similar(positive=pos, negative=neg, topn=(100000000 if fill else topn)))
    sim1 = []
    if yfilter != None:
        done = False
        for idx,s in enumerate(sim0):
            if not done:
                if s[1]<low_thresh:
                    done=True
                else:
                    filtered = len(list(select(_LEXICON, {'lemma':s[0], 'yl':yfilter})['lemma'])) == 0
                    if (not filtered):
                        if (s[1]>=thresh):
                            if (len(sim1)<topn):
                                sim1.append(s)
                            else:
                                done=True
                        else:
                            if (fill and len(sim1)<topn):
                                sim1.append(s)
                            else:
                                done=True
                    elif (fill and len(sim1)<topn):
                        pass
                    else:
                        done=True
            else:
                print('oooo')
                break
        sim0 = sim1
        sim1 = []
    elif thresh>0:
        for s in sim0:
            if s[1]<low_thresh:
                break
            if s[1]>thresh or (fill and s[1]<=thresh and len(sim1)<topn):
                sim1.append(s)
        sim0 = sim1
        sim1 = []

    if comp != -1:
        sim1 = [s[0] if comp==0 else s[1] for s in sim0]
        sim0=sim1
        sim1=[]
    return sim0

def vsim(vector, topn=10, thresh=0, low_thresh=0, comp=-1, model=_MODEL, yfilter=None, fill=None):
    sim = model.wv.similar_by_vector(vector, topn=topn)
    if comp == 0:
        return [s[0] for s in sim]
    elif comp == 1:
        return [s[1] for s in sim]  
    else:
        return sim

def wanal(str_anal, topn=10, thresh=0, low_thresh=0, comp=-1, model=_MODEL, yfilter=None, fill=None):
    p1 = str_anal[:str_anal.index('=')]
    p2 = str_anal[str_anal.index('=')+1:]
    neg = p1[:p1.index(':')].replace(' ', '')
    pos1 = p1[p1.index(':')+1:].replace(' ', '')
    pos2 = p2[:p2.index(':')].replace(' ', '')
    return wsim([pos1,pos2], neg, topn=topn, thresh=thresh, low_thresh=low_thresh, comp=comp, model=model, yfilter=yfilter, fill=fill)

def wdist(w1, w2, model=_MODEL):
    w1 = model.wv[w1] if type(w1) is str else w1
    w2 = model.wv[w2] if type(w2) is str else w2
    return 1-scipy.spatial.distance.cosine(w1, w2)

def nwdist(ws1, ws2, model=_MODEL):
    return model.wv.n_similarity(ws1,ws2)

def wchoesion(words, gamma=1, model=_MODEL):
    sum = 0
    for w1 in words:
        for w2 in words:
            sum = sum + (math.pow((1-wdist(w1,w2,model=model)),gamma) if w1!=w2 else 0)
    return (sum / (math.pow(len(words),2)-len(words))) if (math.pow(len(words),2)-len(words))!=0 else 0

def gety(w, model=_MODEL):
    ys = list(select(_LEXICON, {'lemma':w})['yl'])
    return max(set(ys), key=ys.count)

def is_name(w, model=_MODEL):
    return 'n'==gety(w, model)

def is_adj(w, model=_MODEL):
    return 'j'==gety(w, model)

def is_verb(w, model=_MODEL):
    return 'v'==gety(w, model)

def is_adv(w, model=_MODEL):
    return 'r'==gety(w, model)

def dist_matrix2(words1, city, topic, words2=[], norm=True, show=True, fig_size=_SNS_FIG_SIZE, model=_MODEL):
    matrix=[]
    words2 = words2 if len(words2)>0 else words1
    for w1 in words1:
        row = []
        for w2 in words2:
            f1 = wdist(wdiff('city', city, model=model), wdiff(topic,w1, model=model), model=model)
            f2 = wdist(wdiff('city', city, model=model), wdiff(topic,w2, model=model), model=model)
            f = f1*f2 if (f1>0 and f2>0) else 0
            row.append(wdist(w1,w2,model=model)*f if w1!=w2 else 1)
        matrix.append(row)
    if norm:
        vlen = len(matrix[0])
        flat_matrix = list(np.array(matrix).reshape(-1))
        flat_matrix = normalize(flat_matrix)
        matrix = list(np.array(flat_matrix).reshape(int(len(flat_matrix)/vlen),vlen))
    if show:
        _fig_size = (10,10)
        sns.set(rc={'figure.figsize': fig_size })
        mat=sns.heatmap(np.array(matrix))
    return {'matrix': matrix, 'words': ((words1, words2) if words1 != words2 else words1)}

def dist_matrix3(words1, words2=[], norm=True, gamma=1, show=True, fig_size=_SNS_FIG_SIZE, model=_MODEL):
    matrix=[]
    words2 = words2 if len(words2)>0 else words1
    for w1 in words1:
        row = []
        for w2 in words2:
            row.append(math.pow(wdist(w1,w2,model=model), gamma))
        matrix.append(row)
    if norm:
        vlen = len(matrix[0])
        flat_matrix = list(np.array(matrix).reshape(-1))
        flat_matrix = normalize(flat_matrix)
        matrix = list(np.array(flat_matrix).reshape(int(len(flat_matrix)/vlen),vlen))
    if show:
        _fig_size = (10,10)
        sns.set(rc={'figure.figsize': fig_size })
        mat=sns.heatmap(np.array(matrix))
    return {'matrix': matrix, 'words': ((words1, words2) if words1 != words2 else words1)}

def dist_matrix(words1, words2=[], norm=True, show=True, fig_size=_SNS_FIG_SIZE, model=_MODEL):
    matrix=[]
    words2 = words2 if len(words2)>0 else words1
    for w1 in words1:
        row = []
        for w2 in words2:
            row.append(wdist(w1,w2,model=model))
        matrix.append(row)
    if norm:
        vlen = len(matrix[0])
        flat_matrix = list(np.array(matrix).reshape(-1))
        flat_matrix = normalize(flat_matrix)
        matrix = list(np.array(flat_matrix).reshape(int(len(flat_matrix)/vlen),vlen))
    if show:
        _fig_size = (10,10)
        sns.set(rc={'figure.figsize': fig_size })
        mat=sns.heatmap(np.array(matrix))
    return {'matrix': matrix, 'words': ((words1, words2) if words1 != words2 else words1)}

def sort_dist_matrix(dist_matrix, method='complete', show=True, norm=False, initial_values=[]):
    if norm:
        vlen = len(dist_matrix[0])
        flat_matrix = list(np.array(dist_matrix).reshape(-1))
        flat_matrix = normalize(flat_matrix)
        dist_matrix = list(np.array(flat_matrix).reshape(int(len(flat_matrix)/vlen),vlen))
    if int(dist_matrix[0][0]) == 1:
        dist_matrix = [[(1-v) for v in row] for row in dist_matrix]
    if type(dist_matrix) is not np.array:
        dist_matrix = np.array([np.array(row) for row in dist_matrix])
    out = compute_serial_matrix(dist_matrix, method)
    if show:
        plot_dist_matrix(out['ordered_dist_mat'])
    if initial_values != []:
        out['words'] = [initial_values[i] for i in out['res_order']]
    return out

def plot_dist_matrix(dist_matrix):
    N=len(dist_matrix)
    plt.pcolormesh(dist_matrix)
    plt.colorbar()
    plt.xlim([0,N])
    plt.ylim([0,N])
    plt.show()

def dm_cluster(dist_matrix, elements, eps='auto-75', min_samples=2):
    if 'auto' in eps:
        flat = list(np.reshape(dist_matrix,-1))
        eps = ((sum(flat)-math.sqrt(len(flat))) / (len(flat)-math.sqrt(len(flat)))) - (0 if eps=='auto-50' else np.std(flat))
        eps = eps if eps > 0 else 0.1
    clustering = DBSCAN(eps=eps, min_samples=2, metric='precomputed').fit(dist_matrix)
    out = {
        'core_sample_indices': clustering.core_sample_indices_,
        'components': clustering.components_,
        'labels': list(clustering.labels_),
    }
    cluster = {c:{'words': [], 'score':0} for c in list(set(out['labels']))}
    for i,l in enumerate(out['labels']):
        try:
            cluster[l]['words'].append(elements[i])
        except:
            print(i,l,elements)
    for c in list(cluster.keys()):
        rfreq = len(cluster[c]['words'])/len(elements)
        rfreq_noout = len(cluster[c]['words'])/(len(elements)-(0 if -1 not in cluster else len(cluster[-1]['words']))) if (len(elements)-(0 if -1 not in cluster else len(cluster[-1]['words'])))>0 else 0
        cho_score = wchoesion(cluster[c]['words'], gamma=1)
        cluster[c]['score'] = {'rfreq': rfreq, 'rfreq_noout':rfreq_noout, 'choesion':cho_score}
    out['cluster'] = cluster
    return out

def n_select(df, col, n_val):
    dfout = df.copy()
    if type(n_val) is int or type(n_val) is float:
        n_val = [n_val]
    if type(n_val[0]) is str:
        op = n_val[0]
        n_val = n_val[1:]
        if op == '=':
            dfout = dfout.loc[dfout[col].isin(n_val)].copy()
        elif op == '>':
            dfout = dfout.loc[dfout[col] > n_val[0]].copy()
        elif op == '>=':
            dfout = dfout.loc[dfout[col] >= n_val[0]].copy()
        elif op == '<':
            dfout = dfout.loc[dfout[col] < n_val[0]].copy()
        elif op == '<=':
            dfout = dfout.loc[dfout[col] <= n_val[0]].copy()
        elif op == '!':
            dfout = dfout.loc[~dfout[col].isin(n_val)].copy()
        elif op == 'in':
            dfout = dfout.loc[(dfout[col] >= n_val[0]) & (dfout[col] <= n_val[1])].copy()
        elif op == '!in':
            dfout = dfout.loc[~((dfout[col] >= n_val[0]) & (dfout[col] <= n_val[1]))].copy()
        else:
            print('_semanticF: n_select() from big_select(): operation' + op + 'not valid')
    else:
        dfout = dfout.loc[dfout[col].isin(n_val)].copy()
    return dfout

def o_select(df, col, o_val):
    dfout = df.copy()
    if type(o_val) is str:
        o_val = [o_val]
    if o_val[0] == '!':
        dfout = dfout.loc[~dfout[col].isin(o_val[1:])].copy()
    elif o_val[0] == 're':
        dfout = dfout.loc[dfout[col].str.contains(o_val[1])].copy()
    else:
        dfout = dfout.loc[dfout[col].isin(o_val)].copy()
    return dfout

# big selection on every attributes
def select(df, cts={}):
    dft = dict(df.dtypes)
    dfout = df.copy()
    for ccol in cts:
        t = dft[ccol]
        cval = cts[ccol]
        if t == 'O': # campo stringa
            dfout = o_select(dfout, ccol, cval).copy()
        elif t == 'int64' or t == 'float64': # campo numerico
            dfout = n_select(dfout, ccol, cval).copy()
    return dfout

def normalize(values, method='max-min', exclude_01=False):
    minv = min(values)
    maxv = max(values)
    if exclude_01:
        try:
            minv = min(list(filter(lambda x: x > 0, values)))
            maxv = max(list(filter(lambda x: x < 1, values)))
        except:
            minv = min(values)
            maxv = max(values)
    if method=='max-min':
        return [((v-minv) / (maxv-minv)) if v!=0 and v!=1 else v for v in values]

def discretize_df(df, cols, n, newc=True, norm=False, norm_method=None):
    dfd = df.copy()
    if type(cols) is str:
        cols = [cols]
    for c in cols:
        cname = c if not newc else c+'_discr'
        dfd[cname] = discretize(df[c], n, norm, norm_method)
    return dfd

def discretize(values, n, norm=False, norm_method=None):
    if norm:
        values = normalize(values, norm_method) if norm_method else normalize(values)
    delta = (max(values)-min(values)) / n
    d = [min(values)]
    while(d[-1:][0]<max(values)):
        d.append(d[-1:][0]+delta)
    dis = []
    for v in values:
        i = 0
        while v > d[i+1]:
            i=i+1
        dis.append(i)
    return dis

def seriation(Z,N,cur_index):
    '''
        input:
            - Z is a hierarchical tree (dendrogram)
            - N is the number of points given to the clustering process
            - cur_index is the position in the tree for the recursive traversal
        output:
            - order implied by the hierarchical tree Z

        seriation computes the order implied by a hierarchical tree (dendrogram)
    '''
    if cur_index < N:
        return [cur_index]
    else:
        left = int(Z[cur_index-N,0])
        right = int(Z[cur_index-N,1])
        return (seriation(Z,N,left) + seriation(Z,N,right))

def compute_serial_matrix(dist_mat,method='complete'):
    '''
        input:
            - dist_mat is a distance matrix
            - method = ["ward","single","average","complete"]
        output:
            - seriated_dist is the input dist_mat,
              but with re-ordered rows and columns
              according to the seriation, i.e. the
              order implied by the hierarchical tree
            - res_order is the order implied by
              the hierarhical tree
            - res_linkage is the hierarhical tree (dendrogram)

        compute_serial_matrix transforms a distance matrix into
        a sorted distance matrix according to the order implied
        by the hierarchical tree (dendrogram)
    '''
    N = len(dist_mat)
    flat_dist_mat = squareform(dist_mat)
    res_linkage = linkage(flat_dist_mat, method=method,preserve_input=True)
    res_order = seriation(res_linkage, N, N + N-2)
    seriated_dist = np.zeros((N,N))
    a,b = np.triu_indices(N,k=1)
    seriated_dist[a,b] = dist_mat[ [res_order[i] for i in a], [res_order[j] for j in b]]
    seriated_dist[b,a] = seriated_dist[a,b]

    return {'ordered_dist_mat':seriated_dist, 'res_order':res_order, 'res_linkage':res_linkage}

def pretty(d, indent=0, nokeys=[]):
    for key, value in d.items():
        if key not in nokeys:
            print('\t' * indent + str(key) + ': ' + ('{' if isinstance(value, dict) else ''))
            if isinstance(value, dict):
                pretty(value, indent+1, nokeys=nokeys)
                print('\t' * indent + '}')
            else:
                print('\t' * (indent+1) + str(value))



# get the most similar words to each single entity
# - model: w2v model
# - entities: list of strings
# - total_words: list of strings
# - threshold: minimum similarity, in [0,1]
# - topn: max words for each entity (if -1, shows every word)
# - include_sim: boolean, includes similarity values
def get_related_words(model, entities, total_words, threshold, topn, include_sim=True):
    return {e: _get_related_words_single(model, e, total_words, threshold, topn, include_sim) for e in entities}


def _get_related_words_single(model, entity, total_words, threshold, topn, include_sim):
    # calc similarities
    top_words = {w: 1-model.wv.distance(w, entity) for w in total_words}
    top_words = sort_dict_by_value(top_words, True)

    # check threshold and topn
    if topn==-1:
        topn = len(total_words)
    top_words = {w: top_words[w] for w in list(top_words.keys())[:topn] if top_words[w]>threshold}

    #return
    if include_sim==False:
        return list(top_words.keys())
    else:
        return top_words


# calculate distances between every topic,
# by averaging top 3 similar words for each pair of topics
# returns dict: topic_dist['crime']['vegetation']=0.22
def calc_topic_sim(model):
    topic_sim = {x: {y: 0 for y in C._TOPIC if x != y} for x in C._TOPIC}
    for t1 in topic_sim:
        for t2 in topic_sim[t1]:
            dists = [1-model.wv.distance(k1, k2) for k1 in C._TOPIC[t1] for k2 in C._TOPIC[t2]]
            topic_sim[t1][t2] = np.mean(sorted(dists, reverse=True)[:3])
    return topic_sim


# assumo che topic non definisce una città se ha meno del 15% delle parole mostrate
# fa combinazione lineare di frequenze assolute e relative dei topic rimanenti
def score(f_ass, threshold=0.15):
    f_rel = [x/np.sum(f_ass) for x in f_ass]
    scores = [f_ass[i] * f_rel[i] for i in range(len(f_ass)) if f_rel[i] >= threshold]
    return np.sum(scores) / 10


def flatten(arr):
    return [item for sublist in arr for item in sublist]


def sort_dict_by_key(d, reverse=False):
    return {key: d[key] for key in sorted(d.keys(), reverse=reverse)}


def sort_dict_by_value(d, reverse=True):
    return dict(sorted(d.items(), key=lambda item: item[1], reverse=reverse))

def gcorr(lst, g):
    m = max(lst)
    return [math.pow((v/m),g)*m for v in lst]

def exp(x,e=math.e):
    return math.pow(e,x)

def rescale(lst, NewMin, NewMax):
    OldMin = min(lst)
    OldMax = max(lst)
    return [((((OldValue - OldMin) * (NewMax - NewMin)) / (OldMax - OldMin)) + NewMin) for OldValue in lst]

def rescale_dict(diz, min, max):
    tmp = rescale([v+1 for v in diz.values()],min,max)
    return {k:tmp[i] for i,k in enumerate(diz.keys())}

def gcorr_dict(diz, g):
    tmp = gcorr(list(diz.values()), g)
    return {k:tmp[i] for i,k in enumerate(diz.keys())}

def rgb2hex(rgb_tuple):
    return '#%02x%02x%02x' % rgb_tuple


def plot_wordcloud(words_frequencies, size=(300, 300), fig_size=(5,5), max_font_size=40,
                   background_color='white', mask_path=None, color='topic'):

    if color == 'topic':
        col_func = lambda *args, **kwargs: C._TOPIC_COLOR[C._REVERSE_TOPIC[args[0]]]
    elif color == 'city':
        col_func = lambda *args, **kwargs: C._CITY_COLOR[C._REVERSE_CITIES[args[0]]]
    else:
        col_func = None

    mask = None
    if mask_path is not None:
        mask = np.array(Image.open(mask_path))

    wc = WordCloud(width = size[0],
                   height = size[1],
                   max_font_size = max_font_size,
                   background_color = background_color,
                   color_func = col_func,
                   mask = mask)
    wc.generate_from_frequencies(words_frequencies)

    plt.figure(figsize=fig_size)
    plt.imshow(wc, interpolation="bilinear")
    plt.axis("off")
    plt.figure()
    plt.show()

def describe(values, exclude_0=False, print_res=True, print_plot=True, hue=None, colors=None, over=False, fsize=None, title=''):
    return describe_df(pd.DataFrame(values, columns=['values']), x='values', exclude_0=exclude_0, print_res=print_res, print_plot=print_plot, hue=hue, colors=colors, over=over, fsize=fsize, title=title)

def describe_df(df, x='weight', exclude_0=False, print_res=True, print_plot=True, hue=None, colors=None, over=False, fsize=None, title=''):
    dfx = df[x] if not exclude_0 else select(df, {x:['!',0]})[x]
    drs = scipy.stats.describe(dfx)
    values = pd.Series([np.int64(v) for v in list(dfx)])
    drp = dict(values.describe())
    drt = { 'n obs':  drs[0],
            'values': dfx,
            'mean':  drs[2],
            'variance':  drs[3],
            'std':  drp['std'] if 'std'in list(drp.keys()) else 'NaN',
            'min':  drp['min'] if 'min'in list(drp.keys()) else 'NaN',
            'q25':  drp['25%'] if '25%'in list(drp.keys()) else 'NaN',
            'q50':  drp['50%'] if '50%'in list(drp.keys()) else 'NaN',
            'q75':  drp['75%'] if '75%'in list(drp.keys()) else 'NaN',
            'max':  drp['max'] if 'max'in list(drp.keys()) else 'NaN',
            'range':  drp['max']-drp['min'] if 'max' in list(drp.keys()) else 'NaN',
            'skewness':  drs[4],
            'kurtosis':  drs[5]}
    if print_res:
        print('# Describe variable: ' + x + '\n')
        [print('• ' + r + ' :   ' + str(drt[r])) for r in drt if r !='values']
        print('\n')
    if print_plot:
        if not over:
            plt.figure()
        if hue != None:
            df = df.sort_values(hue)
            values = pd.Series([np.int64(v) for v in list(dfx)])
            hue = [str(v) for v in list(df[hue])]
        # palette = _plot_colorp if not colors else colors
        # fsize = fsize if fsize!=None else _SNS_FIG_SIZE
        # sns.displot(x=values, kde=True, hue=hue, palette=palette, aspect=(fsize[0]/fsize[1]), element="step").set(title=title)
        sns.displot(dfx, kde=True, hue=hue).set(title=title)
        #sns.set(rc={'figure.figsize':_fig_size})
    return drt

def get_questions(filename):
    f = open(filename, 'r')
    q = f.read()
    f.close()
    q = q.split(':')[1:]
    q = [x.split('\n') for x in q]
    q = {x[0].strip(): x[1:] for x in q}
    for k in ['gram3-comparative', 'gram4-superlative', 'gram7-past-tense', 'gram8-plural', 'gram9-plural-verbs']:
        del q[k]
    for k in q:
        q[k] = [[word.lower() for word in row.split(' ')] for row in q[k] if row != '']
    return q


def calc_accuracy(model, q):
    accuracy = []
    topn = 10

    for category in q:
        correct = 0
        count = 0
        for question in q[category]:
            count += 1
            try:
                result = model.wv.most_similar(positive=[question[1], question[2]], negative=[question[0]], topn=topn)
                result = [r[0] for r in result]
                correct += 1 - result.index(question[3]) / topn
            except:
                pass
        accuracy.append([category, correct, count, correct/count])
        print('DONE', category)

    return pd.DataFrame(accuracy, columns=['category', 'correct', 'count', 'ratio'])


def normalize_text(t):
    return unidecode.unidecode(t.lower().replace(' ', '_').replace('-', '_'))


def filter_country(model, df, country, min_pop, max_count, min_sim):
    df = df[(df['country']==country) & (df['population']>=min_pop)]
    cities = [c for c in list(df['name']) if c in model.wv and country in model.wv and
              1-model.wv.distance(country, c) > min_sim]
    return cities[:max_count]


def get_cities(model, min_pop=300000, max_count=15, min_sim=0.15, flat=False, show_coords=False):
    df = pd.read_csv(_PATHS._BASE_DIR + 'DS_Utils\\\\wcities_coords.csv')
    df['coords'] = list(zip(df.lat, df.lng))
    coords = df[['name', 'coords']].set_index('name').to_dict()['coords']

    cities = {c: filter_country(model, df, c, min_pop, max_count, min_sim) for c in set(df['country'])}
    cities = {k: v for k,v in cities.items() if v} #remove empty lists

    if show_coords:
        cities = {k: {city: coords[city] for city in v} for k,v in cities.items()}
    if flat:
        if show_coords:
            cities = {k: v for d in list(cities.values()) for k, v in d.items()}
        else:
            cities = flatten(list(cities.values()))
    return cities
