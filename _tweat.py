from itertools import chain, combinations
from scipy.spatial import distance
import numpy as np


def tweat(embedding, X, Y, A, B, T, verbose=True, calc_pvalue=False):

    if verbose:
        print('X =', X)
        print('Y =', Y)
        print('\nA =', A)
        print('B =', B)
        print('\nT =', T)
        
    eff_size = diff_assoc(embedding, X, Y, A, B, None, None)
    p_value = 'x'
    if calc_pvalue:
        p_value = get_bias_scores_mean_err(embedding, X, Y, A, B, None, None, 2000)
    
    if verbose:
        print('\nSENZA TOPIC:')
        print(' - eff_size =', eff_size)
        print(' - p-value =', p_value)
        
    X2 = [embedding[x] for x in X]
    Y2 = [embedding[y] for y in Y]
    A2 = [embedding[a] + embedding[t] for a in A for t in T]
    B2 = [embedding[b] + embedding[t] for b in B for t in T]

    eff_size_t = diff_assoc(embedding, X2, Y2, A2, B2, None, None)
    topic_impact = eff_size_t - eff_size
    
    if verbose:
        print('\nCON TOPIC:')
        print(' - eff_size =', eff_size_t)
        print('\n >>>> TOPIC IMPACT =', topic_impact)
        
    return topic_impact



def get_most_polarized_attributes(embedding, A, B, min_length):
    psa = [x for x in list(powerset(A)) if len(x) >= min_length]
    psb = [x for x in list(powerset(B)) if len(x) >= min_length]
    
    sims = [[A1, B1, embedding.n_similarity(A1, B1)] for A1 in psa for B1 in psb]
    s2 = [x[2] for x in sims]
    idx = s2.index(min(s2))
    
    return sims[idx][:2]



def powerset(iterable):
    s = list(iterable)
    return [list(x) for x in chain.from_iterable(combinations(s, r) for r in range(len(s)+1))]



def cos_sim(arr1, arr2):
    v1 = np.array(arr1).mean(axis=0)
    v2 = np.array(arr2).mean(axis=0)
    return 1-distance.cosine(v1, v2)

def word_assoc(embedding, w, A, B, T, dist):
    'Calculates difference in mean cosine similarity between a word and two sets of words'
    if T is None: 
        if type(A[0]) is str:
            return embedding.n_similarity([w], A) - embedding.n_similarity([w], B)
        else:
            return cos_sim([w], A) - cos_sim([w], B)
    else: 
        wat = np.mean([dist[w][a][t] for a in A for t in T])
        wbt = np.mean([dist[w][b][t] for b in B for t in T])
        return wat - wbt

def diff_assoc(embedding, X, Y, A, B, T, dist):
    'Calculates the WEAT test statics for four sets of words in an embeddings'
    word_assoc_X = np.array(list(map(lambda x : word_assoc(embedding, x, A, B, T, dist), X)))
    word_assoc_Y = np.array(list(map(lambda y : word_assoc(embedding, y, A, B, T, dist), Y)))
    mean_diff = np.mean(word_assoc_X) - np.mean(word_assoc_Y)
    std = np.std(np.concatenate((word_assoc_X, word_assoc_Y), axis=0))
    return mean_diff / std

def get_bias_scores_mean_err(embedding, X, Y, A, B, T, dist, samples):
    'Caculate the mean WEAT statistic and standard error using a permutation test on the sets of words'
    subset_size_target = len(X) // 2
    subset_size_attr = len(A) // 2
    bias_scores = []
    tot = X + Y
    
    for i in range(samples):
        sX = np.random.choice(tot, subset_size_target, replace=False)
        sY = [v for v in tot if v not in sX]
        bias_scores.append(diff_assoc(embedding, sX, sY, A, B, T, dist))
        
    total_score = diff_assoc(embedding, X, Y, A, B, T, dist)
    if total_score < 0:
        res = np.sum([1 for score in bias_scores if score < total_score])/samples
    else:
        res = np.sum([1 for score in bias_scores if score > total_score])/samples
    return res