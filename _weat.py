import numpy as np
from . import _semanticF as s
from sympy.utilities.iterables import multiset_permutations

def unit_vector(vec):
    """
    Returns unit vector
    """
    return vec / np.linalg.norm(vec)


def cos_sim(v1, v2):
    """
    Returns cosine of the angle between two vectors
    """
    v1_u = unit_vector(v1)
    v2_u = unit_vector(v2)
    return np.clip(np.tensordot(v1_u, v2_u, axes=(-1, -1)), -1.0, 1.0)


def weat_association(W, A, B):
    """
    Returns association of the word w in W with the attribute for WEAT score.
    s(w, A, B)
    :param W: target words' vector representations
    :param A: attribute words' vector representations
    :param B: attribute words' vector representations
    :return: (len(W), ) shaped numpy ndarray. each rows represent association of the word w in W
    """
    return np.mean(cos_sim(W, A), axis=-1) - np.mean(cos_sim(W, B), axis=-1)


def weat_differential_association(X, Y, A, B):
    """
    Returns differential association of two sets of target words with the attribute for WEAT score.
    s(X, Y, A, B)
    :param X: target words' vector representations
    :param Y: target words' vector representations
    :param A: attribute words' vector representations
    :param B: attribute words' vector representations
    :return: differential association (float value)
    """
    return np.sum(weat_association(X, A, B)) - np.sum(weat_association(Y, A, B))


def weat_p_value(X, Y, A, B):
    """
    Returns one-sided p-value of the permutation test for WEAT score
    CAUTION: this function is not appropriately implemented, so it runs very slowly
    :param X: target words' vector representations
    :param Y: target words' vector representations
    :param A: attribute words' vector representations
    :param B: attribute words' vector representations
    :return: p-value (float value)
    """
    diff_association = weat_differential_association(X, Y, A, B)
    target_words = np.concatenate((X, Y), axis=0)

    # get all the partitions of X union Y into two sets of equal size.
    idx = np.zeros(len(target_words))
    idx[:len(target_words) // 2] = 1

    partition_diff_association = []
    for i in multiset_permutations(idx):
        i = np.array(i, dtype=np.int32)
        partition_X = target_words[i]
        partition_Y = target_words[1 - i]
        partition_diff_association.append(weat_differential_association(partition_X, partition_Y, A, B))

    partition_diff_association = np.array(partition_diff_association)

    return np.sum(partition_diff_association > diff_association) / len(partition_diff_association)


def weat_score(X, Y, A, B):
    """
    Returns WEAT score
    X, Y, A, B must be (len(words), dim) shaped numpy ndarray
    CAUTION: this function assumes that there's no intersection word between X and Y
    :param X: target words' vector representations
    :param Y: target words' vector representations
    :param A: attribute words' vector representations
    :param B: attribute words' vector representations
    :return: WEAT score
    """

    x_association = weat_association(X, A, B)
    y_association = weat_association(Y, A, B)


    tmp1 = np.mean(x_association, axis=-1) - np.mean(y_association, axis=-1)
    tmp2 = np.std(np.concatenate((x_association, y_association), axis=0))

    return tmp1 / tmp2


def wefat_p_value(W, A, B):
    """
    Returns WEFAT p-value
    W, A, B must be (len(words), dim) shaped numpy ndarray
    CAUTION: not implemented yet
    :param W: target words' vector representations
    :param A: attribute words' vector representations
    :param B: attribute words' vector representations
    :return: WEFAT p-value
    """
    pass


def wefat_score(W, A, B):
    """
    Returns WEFAT score
    W, A, B must be (len(words), dim) shaped numpy ndarray
    CAUTION: this function assumes that there's no intersection word between A and B
    :param W: target words' vector representations
    :param A: attribute words' vector representations
    :param B: attribute words' vector representations
    :return: WEFAT score
    """
    tmp1 = weat_association(W, A, B)
    tmp2 = np.std(np.concatenate((cos_sim(W, A), cos_sim(W, B)), axis=0))

    return np.mean(tmp1 / tmp2)

def balance_word_vectors(A, B):
    """
    Balance size of two lists of word vectors by randomly deleting some vectors in larger one.
    If there are words that did not occur in the corpus, some words will ignored in get_word_vectors.
    So result word vectors' size can be unbalanced.
    :param A: (len(words), dim) shaped numpy ndarrary which is word vectors
    :param B: (len(words), dim) shaped numpy ndarrary which is word vectors
    :return: tuple of two balanced word vectors
    """

    diff = len(A) - len(B)

    if diff > 0:
        A = np.delete(A, np.random.choice(len(A), diff, 0), axis=0)
    else:
        B = np.delete(B, np.random.choice(len(B), -diff, 0), axis=0)

    return A, B

def weat_m1(X_key, Y_key, Topic, Subject, Positive, Negative, AB_topn=10, p_value=False, verbose=True):
    pos_analogy = Topic + ':' + Positive + '=' + Subject + ':x'
    neg_analogy = Topic + ':' + Negative + '=' + Subject + ':x'

    A_key = s.wanal(pos_analogy, comp=0)[:AB_topn]
    B_key = s.wanal(neg_analogy, comp=0)[:AB_topn]

    X = [s.wvec(word) for word in X_key]
    Y = [s.wvec(word) for word in Y_key]
    A = [s.wvec(word) for word in A_key]
    B = [s.wvec(word) for word in B_key]    

    X, Y = balance_word_vectors(X, Y)
    A, B = balance_word_vectors(A, B)

    score = weat_score(X, Y, A, B)
    their_p_value = weat_p_value(X, Y, A, B) if p_value else False
    our_p_value = get_bias_scores_mean_err(X_key, Y_key, A_key, B_key)

    if verbose:
        print('• pos_analogy:', pos_analogy)
        print('• neg_analogy:', neg_analogy)
        print()
        print('• X_key:', X_key)
        print('• Y_key:', Y_key)
        print()
        print('• A_key:', A_key)
        print('• B_key:', B_key)
        print()
        print('→ score:', score)
        print('→ their_p_value:', their_p_value)
        print('→ our_p_value:', our_p_value)

    return {
        'pos_analogy': pos_analogy,
        'neg_analogy': neg_analogy,
        'X_key': X_key,
        'Y_key': Y_key,
        'A_key': A_key,
        'B_key': B_key,
        'score': score,
        'their_p_value': their_p_value,
        'our_p_value': our_p_value,
    }


def weat_m2(topic_words, X_city, Y_city, Topic, Positive, Negative, thresh=0.35, p_value=False, verbose=True):
    X_key = []
    Y_key = []
    [X_key.extend(list(set([word[0] for word in topic_words[cx][Topic] if word[1]>0.3]))) for cx in X_city]
    [Y_key.extend(list(set([word[0] for word in topic_words[cy][Topic] if word[1]>0.3]))) for cy in Y_city]

    A_key = s.vsim(np.sum([s.wvec(pw) for pw in Positive], axis=0), comp=0)[:10]
    B_key = s.vsim(np.sum([s.wvec(nw) for nw in Negative], axis=0), comp=0)[:10]

    X = [s.wvec(word) for word in X_key]
    Y = [s.wvec(word) for word in Y_key]
    A = [s.wvec(word) for word in A_key]
    B = [s.wvec(word) for word in B_key]

    X, Y = balance_word_vectors(X, Y)
    A, B = balance_word_vectors(A, B)

    score = weat_score(X, Y, A, B)
    their_p_value = weat_p_value(X, Y, A, B) if p_value else False
    our_p_value = get_bias_scores_mean_err(X_key, Y_key, A_key, B_key, samples=500)

    if verbose:
        print('• pos_analogy:', Positive)
        print('• neg_analogy:', Negative)
        print()
        print('• X_key:', X_key)
        print('• Y_key:', Y_key)
        print()
        print('• A_key:', A_key)
        print('• B_key:', B_key)
        print()
        print('→ score:', score)
        print('→ their_p_value:', their_p_value)
        print('→ our_p_value:', our_p_value)

    return {
        'positive': Positive,
        'negative': Negative,
        'X_key': X_key,
        'Y_key': Y_key,
        'A_key': A_key,
        'B_key': B_key,
        'score': score,
        'their_p_value': their_p_value,
        'our_p_value': our_p_value,
    }


def get_bias_scores_mean_err(X, Y, A, B, T=None, dist=None, samples=2000):
    'Caculate the mean WEAT statistic and standard error using a permutation test on the sets of words'
    subset_size_target = len(X) // 2
    subset_size_attr = len(A) // 2
    bias_scores = []
    tot = X + Y
    
    for i in range(samples):
        sX = np.random.choice(tot, subset_size_target, replace=False)
        sY = [v for v in tot if v not in sX]
        bias_scores.append(diff_assoc(sX, sY, A, B, T, dist))
        
    total_score = diff_assoc(X, Y, A, B, T, dist)
    if total_score < 0:
        res = np.sum([1 for score in bias_scores if score < total_score])/samples
    else:
        res = np.sum([1 for score in bias_scores if score > total_score])/samples
    return res

def diff_assoc(X, Y, A, B, T=None, dist=None):
    'Calculates the WEAT test statics for four sets of words in an embeddings'
    word_assoc_X = np.array(list(map(lambda x : word_assoc(x, A, B, T, dist), X)))
    word_assoc_Y = np.array(list(map(lambda y : word_assoc(y, A, B, T, dist), Y)))
    mean_diff = np.mean(word_assoc_X) - np.mean(word_assoc_Y)
    std = np.std(np.concatenate((word_assoc_X, word_assoc_Y), axis=0))
    return mean_diff / std

def word_assoc(w, A, B, T=None, dist=None):
    'Calculates difference in mean cosine similarity between a word and two sets of words'
    if T is None: 
        if type(A[0]) is str:
            return s._MODEL.wv.n_similarity([w], A) - s._MODEL.wv.n_similarity([w], B)
        else:
            return cos_sim2([w], A) - cos_sim2([w], B)
    else: 
        wat = np.mean([dist[w][a][t] for a in A for t in T])
        wbt = np.mean([dist[w][b][t] for b in B for t in T])
        return wat - wbt

def cos_sim2(arr1, arr2):
    v1 = np.array(arr1).mean(axis=0)
    v2 = np.array(arr2).mean(axis=0)
    return s.wdist(v1, v2)