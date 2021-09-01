# Git repository link
# https://github.com/PimpMyGit/DS_Utils

import os
import io
import re
import math
import glob
import scipy
import pickle
import zipfile
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from fastcluster import linkage
from termcolor import colored
from gensim.models import Word2Vec
from gensim.models.phrases import Phrases, Phraser
from scipy.spatial.distance import pdist, squareform
from . import _paths as _PATHS
#%matplotlib inline


# THIS MODULE NAME
_MODULE_NAME = '_semanticF'
# DIRECTORIES - PROJECT FOLDER and COCA ARCHIVE
_BASE_DIR = _PATHS._BASE_DIR
_COCA_PATH = _PATHS._COCA_PATH
_MODEL_PATH = _BASE_DIR + 'grosso_SG_lemmatize.pkl'
_LEXICON_PATH = _BASE_DIR + '\\Lexicon\\lexicon_lemmatize.csv'
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
    path = _LEXICON_PATH
    lex = pd.read_csv(path, sep='\t')
    if drop_na:
        lex = lex.dropna()
    return lex
_LEXICON = load_lexicon()
# W2V MODEL
_MODEL = pickle.load(open(_MODEL_PATH, 'rb'))
# HEADER GENERALI DEI DATAFRAME
_WLP_HEADER = ['original', 'standard', 'y']
# Y-TAG VARI
_STOP_TAG = 'y_stop'
_LEMMA_TAG = '^n|^jj|^v|^r'
_NEGATION_TAG = '^xx'
_SYMBOLS_TAG = ['y','ge']
# SEABORN
_SNS_FIG_SIZE=_fig_size = (10,10)
sns.set(rc={'figure.figsize':_SNS_FIG_SIZE})

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
        for file in archive.namelist():
            if file in files:
                content[file] = self.read_inzip_file(archive, file)
        return content

    def read_inzip_file(self, archive, file):
        with archive.open(file) as f:
            return f.read()

    def pickle_save(self, filename, object, subfolder='models'):
        path = ''
        if '\\' in filename:
            path = filename
        else:
            if not os.path.exists(_BASE_DIR + '\\' + subfolder + '\\'):
                os.mkdir(_BASE_DIR + '\\' + subfolder + '\\')
            path = _BASE_DIR + '\\' + subfolder + '\\' + filename
        file = open(path + '.pkl', 'wb')
        pickle.dump(object, file)
        file.close()
        self.throw_msg('done', 'object saved in ' + path, 'pickle_save')


    def pickle_load(self, filename, subfolder='models'):
        path = filename if '\\' in filename else _BASE_DIR + '\\' + subfolder + '\\' + filename
        file = open(path + '.pkl', 'rb')
        object = pickle.load(file)
        file.close()
        return object

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
            print(throw_msg('error', 'message category muste be in ' + ', '.join(list(category_colour.keys())), throw_msg.__name__))

#------------------------------------------------------------------------------#

U = Utils()

def get_files(category, type='text'):
    archive = zipfile.ZipFile(_COCA_PATH, 'r')
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
        df.to_csv(_BASE_DIR+filename+'.csv', sep='\t', index=False)
    return df

def massivo_csv(type='wlp'):
    tutti = []
    for category in list(_DICT_CATEGORIES.keys()):
        if category=='Blogs':
            for n in range(1,35):
                tutti.append(_BASE_DIR + '\\' + category + '\\' + type + '_' + _DICT_CATEGORIES[category.capitalize()] + '_' + str(n).zfill(2) + '.csv')
        else:
            for year in range(1990,2020):
                tutti.append(_BASE_DIR + '\\' + category + '\\' + type + '_' + _DICT_CATEGORIES[category.capitalize()] + '_' + str(year) + '.csv')
    return tutti

# def load_lexicon(drop_na=True):
#     path = _LEXICON_PATH
#     lex = pd.read_csv(path, sep='\t')
#     if drop_na:
#         lex = lex.dropna()
#     return lex

def load_csv(category, year, type='wlp', drop_na=True):
    path = _BASE_DIR + '\\' + category + '\\' + type + '_' + _DICT_CATEGORIES[category.capitalize()] + '_' + str(year) + '.csv'
    df = pd.read_csv(path, sep='\t')
    if drop_na:
        df = df.dropna()
    return df

def set_y_stop(df, standard_col='standard', y_col='y', stop_tag=_STOP_TAG):
    y_cols = [y_col] if type(y_col) is str else y_col
    for y_col in y_cols:
        df[y_col] = df.apply(lambda row: stop_tag if row[y_col] in _SYMBOLS_TAG and row[standard_col]=='.' else row[y_col], axis=1)
    return df

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
    return select(df.dropna(), {lemma_col:['re', tag+'|y_stop']})

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

def get_sentences(dfs, sentence_col='standard', sep='.', preprocess=None):
    sentences = []
    dfs = [dfs] if type(dfs) is not list else dfs
    if preprocess:
        print('starting df preprocess ...')
        dfs = [preprocess_df(df, preprocess) for df in dfs]
    for df in dfs:
        text = ' '.join(list(df[sentence_col]))
        sentences = sentences + [[word.replace('-', '_') for word in sentence.split()] for sentence in text.split(' . ')]
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
    preprocess=None, min_count=5, vector_size=300,
    model='SG', model_filename='default_model'):
    print('getting sentences ...')
    sentences = get_sentences(dfs, sentence_col=sentence_col, sep=sep, preprocess=preprocess)
    print('done sentences!')
    print('getting bi-grams ...')
    bi_gram_sentences = generate_bi_grams(sentences,
                                        min_count=5,
                                        threshold=7,
                                        progress_per=1000,
                                        #scoring='npmi'
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

def wvec(w, model=_MODEL):
    return model.wv[w]

def wsum(w1, w2, model=_MODEL):
    w1 = model.wv[w1] if type(w1) is str else w1
    w2 = model.wv[w2] if type(w2) is str else w2
    return w1 + w1

def wnorm(w, model=_MODEL):
    vec = model.wv[w] if type(w) is str else w
    return math.sqrt(sum([math.pow(v,2) for v in vec]))

def wsim(pos, neg=[], topn=10, thresh=0, comp=-1, model=_MODEL, yfilter=None, fill=None):
    fill = True if (thresh==0 and fill==None) else fill
    sim0 = list(model.wv.most_similar(positive=pos, negative=neg, topn=(100000000 if fill else topn)))
    sim1 = []
    if yfilter != None:
        for s in sim0:
            filtered = len(list(select(_LEXICON, {'lemma':s[0], 'yl':yfilter})['lemma'])) == 0
            if (not filtered) and ((s[1]>thresh) or (fill and len(sim1)<topn)):
                sim1.append(s)
            elif (fill and len(sim1)<topn):
                pass
            else:
                break
        sim0 = sim1
        sim1 = []
    elif thresh>0:
        for s in sim0:
            if s[1]>thresh or (fill and s[1]<=thresh and len(sim1)<topn):
                sim1.append(s)
        sim0 = sim1
        sim1 = []

    if comp != -1:
        sim1 = [s[0] if comp==0 else s[1] for s in sim0]
        sim0=sim1
        sim1=[]
    return sim0

def wdist(w1, w2, model=_MODEL):
    w1 = model.wv[w1] if type(w1) is str else w1
    w2 = model.wv[w2] if type(w2) is str else w2
    return 1-scipy.spatial.distance.cosine(w1, w2)

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

def sort_dist_matrix(dist_matrix, method='complete', initial_values=[]):
    if int(dist_matrix[0][0]) == 1:
        dist_matrix = [[(1-v) for v in row] for row in dist_matrix]
    if type(dist_matrix) is not np.array:
        dist_matrix = np.array([np.array(row) for row in dist_matrix])
    out = compute_serial_matrix(dist_matrix, method)
    N=len(out['ordered_dist_mat'])
    plt.pcolormesh(out['ordered_dist_mat'])
    plt.colorbar()
    plt.xlim([0,N])
    plt.ylim([0,N])
    plt.show()
    if initial_values != []:
        out['words'] = [initial_values[i] for i in out['res_order']]
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

def normalize(values , method='max-min'):
    minv = min(values)
    maxv = max(values)
    if method=='max-min':
        return [((v-minv) / (maxv-minv)) for v in values]

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
