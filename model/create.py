# construct final model (see: Notes) and pickle it;
# additionally, make a new data file for new "midrange"
# reviews (i.e., the things we're going to be interested
# in recommending to users)

# suppress .pyc
import sys 
sys.dont_write_bytecode = True 

# model building materials
import nlp_processing as nlpp
from sklearn.linear_model import LogisticRegression
from map_scale import *

# standard/storing
import pandas as pd
import numpy as np
import cPickle
import sqlalchemy
import psycopg2

if __name__ == '__main__':
    
    print 'reading data in and splitting into training and test sets...'
    df = pd.read_csv('data/final_p4k_pre_classifier.csv')
    save_content = df['Content'].copy()
    df['Content'] = df.apply(lambda x: nlpp.remove(x['Artist'], x['Content']), axis=1)
    mid, neg, pos = nlpp.produce_sentiment_data(df)

    neg_label = pd.DataFrame([0] * neg.shape[0])
    pos_label = pd.DataFrame([1] * pos.shape[0])
    train = pd.concat([neg, pos])
    label = pd.concat([neg_label, pos_label])

    print 'building TF-IDF...'
    tfidf = nlpp.build_cv_or_tfidf(neg['Content'], pos['Content'], 'TFIDF', { 'ngram_range': (1, 3), 'min_df': 9 }, None)
    df['Content'] = save_content
    train_transformed = tfidf.fit_transform(train['Content'])

    print 'training Logistic Regression...'
    logreg = LogisticRegression(penalty='l2', C=100.0)
    logreg.fit(train_transformed, label.values.ravel())

    print 'preparing data for final export to CSV and SQL...'
    mid['Mid?'] = 1 # label mid separately since these are the reviews
                    # we're interested in scoring and recommending
    neg['Mid?'] = 0
    pos['Mid?'] = 0

    df = pd.concat([mid, neg, pos])
    text_df = tfidf.transform(df['Content'])
    df['MY_pos'] = logreg.predict_proba(text_df)[:, 1]

    print 'scaling all sentiment analysis scores to 0-10 range...'
    df['NLTK_score'] = df['NLTK_pos'].apply(lambda x: scale_score(x, 0, 1))
    df['TB_score']   = df['TB_pos'].apply(lambda x: scale_score(x, -1, 1))
    df['MY_score']   = df['MY_pos'].apply(lambda x: scale_score(x, 0, 1))

    print 'reordering sentiment analysis scores to fit Pitchfork scoring distribution...'
    score = df[['Score']].sort('Score').values

    df = df.sort('NLTK_score')
    df['NLTK_scaled'] = score.copy()
    df = df.sort('TB_score')
    df['TB_scaled'] = score.copy() 
    df = df.sort('MY_score')
    df['MY_scaled'] = score.copy()

    print 'producing scoring map for Pitchfork score and condensed data representation for my score...'
    p4k_scoring_map = produce_scoring_map(df, 'Score')
    my_score_scale = produce_scoring_map(df, 'MY_score').values()

    print 'saving data to csv -- this is our final copy...'
    df.to_csv('data/final_p4k.csv', index=False)

    print 'saving data to SQL -- this is going to form the foundation of our web app...'
    engine = sqlalchemy.create_engine('postgresql://postgres:@localhost/pitchfork')
    df.to_sql('review', engine, if_exists='replace')

    print 'pickling TF-IDF and Logistic Regression model...'
    cPickle.dump((tfidf, logreg, p4k_scoring_map, my_score_scale), open('tfidf_logreg_maps.pkl', 'w'))




