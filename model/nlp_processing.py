# suppress .pyc
import sys 
sys.dont_write_bytecode = True

# standard
import pandas as pd
import numpy as np

# text extraction/featurization
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer

from nltk.tokenize import wordpunct_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.tag import pos_tag
from nltk.corpus import stopwords

stop = stopwords.words('english')

import re

# to incorporate Lemmatization (unless Lemmatization gives you a large
# performance boost, I would avoid it -- the NLTK library takes a long, long,
# long time)
class LemmaTokenizer(object):
    def __init__(self):
        self.wnl = WordNetLemmatizer()
    def __call__(self, doc):
        return [self.wnl.lemmatize(t) for t in wordpunct_tokenize(doc)]

# to incorporate P.O.S. tagging (again, quite slow)
class POSTokenizer(object):
    def __init__(self):
        self.pt = pos_tag
    def __call__(self, doc):
        return [t[0] + t[1] for t in self.pt(wordpunct_tokenize(doc))]


def produce_sentiment_data(in_df, neg_thresh=6.0, pos_thresh=8.0, reviewer='All', genre='All'):

    '''This function is used to produce positive and negative data sets (i.e., DataFrames)
    that are going to be used in model-building. The reviewer and genre fields are 
    optional, but depending on the circumstances, can be useful to gain deeper
    understanding into a specific critic's writing/feelings about a particular
    genre'''

    df = in_df.copy()
    if reviewer != 'All':
        df = df[df['Reviewer'] == reviewer]
    if genre != 'All':
        df = df[df['Genre'] == genre]

    neg = df['Score'] <= neg_thresh
    pos = df['Score'] >= pos_thresh

    return df[~(neg | pos)], df[neg], df[pos] 


def build_cv_or_tfidf(in_neg_df, in_pos_df, cv_or_tfidf='CV', nlp_params={}, info_thresh=None):

    # get data ready
    neg_df = in_neg_df.copy()
    pos_df = in_pos_df.copy()
    all_content = pd.concat([pos_df, neg_df])

    # fit CountVectorizer
    if cv_or_tfidf == 'CV':
        cv_or_tf = CountVectorizer(**nlp_params)
    else:
        cv_or_tf = TfidfVectorizer(**nlp_params)
    cv_or_tf.fit(all_content)

    # including this because filtering based on information threshold 
    # becomes too computationally punishing with larger ngrams
    if 'ngram_range' in nlp_params and nlp_params['ngram_range'] != (1, 1):
        return cv_or_tf

    # transform positive and negative content
    neg_cv = pd.DataFrame(cv_or_tf.transform(neg_df).todense(), columns=cv_or_tf.vocabulary_)
    pos_cv = pd.DataFrame(cv_or_tf.transform(pos_df).todense(), columns=cv_or_tf.vocabulary_)

    # count up words
    neg_counts = np.sum(neg_cv, axis=0)
    neg_counts = neg_counts + 1 # to avoid division by 0 when calculating ratios (below)
    pos_counts = np.sum(pos_cv, axis=0)
    pos_counts = pos_counts + 1

    all_counts = pd.DataFrame(neg_counts, columns=['Negative'])
    all_counts['Positive'] = pos_counts

    # calculate ratios
    all_counts['Neg_Pos'] = (1.0 * all_counts['Negative'])/all_counts['Positive']
    all_counts['Pos_Neg'] = (1.0 * all_counts['Positive'])/all_counts['Negative']

    if info_thresh is not None:
        all_counts = all_counts[(all_counts['Neg_Pos'] >= info_thresh) | (all_counts['Pos_Neg'] >= info_thresh)]

    # remake CV/TF-IDF with pruned vocabulary (this could be its own function)
    nlp_params['vocabulary'] = all_counts.index 
    if cv_or_tfidf == 'CV':
        cv_or_tf = CountVectorizer(**nlp_params) 
    else:
        cv_or_tf = TfidfVectorizer(**nlp_params)

    return cv_or_tf

# to remove artists from their own reviews
def remove_artist_from_content(artist, content):

    return re.sub(artist, r'', content)

def remove(artist, content):
    try:
        return remove_artist_from_content(artist, content)
    except:
        return ''

def find_significant_words(tfidf_vocab, logreg_coefs, feature_vector, best_or_worst='best'):
    
    ordered_feats = np.argsort(feature_vector)
    ordered_feats = np.array(ordered_feats).ravel()

    result = []

    if best_or_worst == 'best':
        for i in ordered_feats[::-1]:
            if logreg_coefs[i] > 5 and tfidf_vocab[i] not in stop:
                result.append(tfidf_vocab[i])
            if len(result) > 10:
                break
    else:
        for i in ordered_feats:
            if logreg_coefs[i] < -5 and tfidf_vocab[i] not in stop:
                result.append(tfidf_vocab[i])
            if len(result) > 10:
                break

    return result




