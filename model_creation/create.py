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

# standard/storing
import pandas as pd
import cPickle
import sqlalchemy
import psycopg2


def scale_score(x, from_scale_low=-1, from_scale_high=1, to_scale_low=0, to_scale_high=10):

	if from_scale_low < 0:
		temp = from_scale_low

		x -= temp
		from_scale_low -= temp
		from_scale_high -= temp 

	from_scale = (1.0 * x) / (from_scale_high - from_scale_low)
	return (to_scale_high - to_scale_low) * from_scale


if __name__ == '__main__':
	
	print 'reading data in and splitting into training and test sets...'
	df = pd.read_csv('data/final_p4k_pre_classifier.csv')
	mid, neg, pos = nlpp.produce_sentiment_data(df)

	neg_label = pd.DataFrame([0] * neg.shape[0])
	pos_label = pd.DataFrame([1] * pos.shape[0])
	train = pd.concat([neg, pos])
	label = pd.concat([neg_label, pos_label])

	print 'building TF-IDF...'
	tfidf = nlpp.build_cv_or_tfidf(neg, pos, 'TFIDF', { 'ngram_range': (1, 3), 'min_df': 9 }, None)
	train_transformed = tfidf.fit_transform(train['Content'])

	print 'training Logistic Regression...'
	logreg = LogisticRegression(penalty='l2', C=100.0)
	logreg.fit(train_transformed, label.values.ravel())

	print 'pickling TF-IDF and Logistic Regression model...'
	cPickle.dump((tfidf, logreg), open('tfidf_logreg.pkl', 'w'))

	print 'preparing data for final export to CSV and SQL...'
	mid['Mid?'] = 1 # label mid separately since these are the reviews
					# we're interested in scoring and recommending
	neg['Mid?'] = 0
	pos['Mid?'] = 0

	df = pd.concat([mid, neg, pos])
	text_df = tfidf.transform(df['Content'])
	df['MY_pos'] = logreg.predict_proba(text_df)[:, 1]

	print 'scaling all sentiment analysis scores'
	df['NLTK_score'] = df['NLTK_pos'].apply(lambda x: scale_score(x, 0, 1))
	df['TB_score'] 	 = df['TB_pos'].apply(lambda x: scale_score(x, -1, 1))
	df['MY_score'] 	 = df['MY_pos'].apply(lambda x: scale_score(x, 0, 1))

	print 'saving data to csv -- this is our final copy...'
	df.to_csv('data/final_p4k.csv', index=False)

	print 'saving data to SQL -- this is going to form the foundation of our web app...'
	engine = sqlalchemy.create_engine('postgresql://postgres:@localhost/pitchfork')
	df.to_sql('review', engine)





