# This .py file isn't a module so much as it is a workhorse:
# Its primary goal is to find the best version of the following data pipeline:
	# (A) CV and TF-IDF (i.e., tuning minimum document frequency, ngram range, etc.)
	# (B) "Importance Threshold" (i.e., to be included as a feature, a word or ngram
	# 	   must show up in far more 'positive' documents than 'negative' documents, 
	#	   or vice versa. The "far more" is the threshold of interest.
	# (C) Naive Bayes OR Logistic Regression (including tuning parameters)

# my sentiment analysis tools
import sentiment_analysis as sa 

# for cross-validation
from sklearn.cross_validation import train_test_split
from sklearn import metrics

# standard
import pandas as pd

# machine learning
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression

# to test (this is a relatively complex data pipeline) 
# (See: Combinatorial Explosion) 
MODELS = { 
			LogisticRegression: { 'penalty': 	['l1', 'l2'],
								  'C':			[0.001, 0.01, 0.1, 1.0, 10.0] },
			MultinomialNB: 		{ 'alpha': 		[0.001, 0.01, 0.1, 1.0, 10.0] }
}
PRODUCE_VOCAB = { 
			cv_or_tfidf: 		[ 'CV', 'TF-IDF' ],
			info_thresh: 		[ None, 1.5, 2.0, 3.0, 4.0, 5.0 ] 
}
NLP_PARAMS = { 
			tokenizer: 			[ None, sa.LemmaTokenizer() ],
			ngram_range: 		[ (1, 1), (1, 2), (1, 3), (1, 4), (1, 5), 
								  (2, 3), (2, 4), (2, 5), (3, 5) ],
			min_df: 			[ 1, 2, 3, 5, 7, 10 ] 
}

if __name__ == '__main__':
	
	df = pd.read_csv(filepath_or_buffer) 
