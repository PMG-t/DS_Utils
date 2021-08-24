import os
import io
import re
import csv
import glob
import zipfile
import pandas as pd
from . import _paths as _PATHS

# COCA - directories - !!!! da aggiustare con le proprie !!!!
_BASE_DIR = _PATHS._BASE_DIR # 'C:\\Users\\tommaso\\Documents\\_Bicocca\\Magistrale\\Data Semantics\\SemanticCities\\'
_COCA_PATH = _PATHS._COCA_PATH # 'C:\\COCA.zip'
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
# HEADER GENERALI DEI FILE
_WLP_HEADER = ['original', 'standard', 'y']
# Y-TAG VARI
_STOP_TAG = 'y_stop'
_LEMMA_TAG = '^nn|^jj|^v|^r'
_NEGATION_TAG = '^xx'
_SYMBOLS_TAG = ['y','ge']


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

def load_csv(category, year, type='wlp', drop_na=True):
    path = _BASE_DIR + '\\' + category + '\\' + type + '_' + _DICT_CATEGORIES[category.capitalize()] + '_' + str(year) + '.csv'
    df = pd.read_csv(path, sep='\t')
    if drop_na:
        df = df.dropna()
    return df

def load_lexicon(drop_na=True):
    path = _BASE_DIR + '\\Lexicon\\lexicon.csv'
    lex = pd.read_csv(path, sep='\t')
    if drop_na:
        lex = lex.dropna()
    return lex

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
    return select(df.dropna(), {lemma_col:['re', tag+'|y_stop']})

def del_symbols_df(df, lemma_col='first_y', tag=_SYMBOLS_TAG):
    tag = [tag] if type(tag) is str else tag
    return select(df.dropna(), {lemma_col:['!']+tag})

def texts_to_sentences(texts):
    total = []
    for text in texts:
        lines = [l.split('\t') for l in text.split('\n')]
        standardized = [x[2] for x in lines if len(x) > 2] # prende solo seconda colonna dal file
        new_text = ' '.join(standardized) # rimette tutto in formato testuale
        sentences = [beautify(x.split()) for x in new_text.split('.')] # divide frasi e le sistema
        total += sentences
    return total

def beautify(text):
    stoplist = {'', '.', ',', ';', ':', '...', '<p>', '\"', '\'', '-'}
    return [word.lower() for word in text if word not in stoplist]

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
            print('_energyF: n_select() from big_select(): operation' + op + 'not valid')
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
