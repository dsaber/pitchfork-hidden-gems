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
import sklearn.cross_validation as cross_validation 
from sklearn import metrics

# standard
import pandas as pd
import numpy as np 
import itertools 

# machine learning
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression

# to store
import cPickle


# to test (this is a relatively complex data pipeline) 
# (See: Combinatorial Explosion) 
PRODUCE_VOCAB = 	  { 
							'cv_or_tfidf': 		[ 'CV', 'TF-IDF' ],
							'info_thresh': 		[ None, 1.5, 2.0, 3.0, 4.0, 5.0 ] 
}
NLP_PARAMS = 		  { 
							'tokenizer': 		[ None, sa.LemmaTokenizer() ],
							'ngram_range': 		[ (1, 1), (1, 2), (1, 3), (1, 4), (1, 5), 
								  				(2, 3), (2, 4), (2, 5), (3, 5) ],
							'min_df': 			[ 1, 2, 3, 5, 7, 10 ] 
}
NAMES = 			  { 
							LogisticRegression: 'LogisticRegression',
							MultinomialNB:		'MultinomialNB'
}
MODELS = 			  { 
							LogisticRegression: { 'penalty': 		['l1', 'l2'],
								  						'C':		[0.001, 0.01, 0.1, 1.0, 10.0] },
							MultinomialNB: 		{ 	'alpha': 		[0.001, 0.01, 0.1, 1.0, 10.0] }
}


# This is essentially a gigantic grid search
def main(scoring_func=metrics.roc_auc_score): 

	result = { } 
	
	df = pd.read_csv('data/final_p4k.csv') 
	mid, neg, pos = sa.produce_sentiment_data(df)
	neg_label = pd.DataFrame([0] * neg.shape[0])
	pos_label = pd.DataFrame([1] * pos.shape[0]) 

	# prepare for cross validation
	neg_kf = cross_validation.KFold(neg.shape[0], n_folds=5, shuffle=True)
	pos_kf = cross_validation.KFold(pos.shape[0], n_folds=5, shuffle=True)

	# loop through train/test sets for any given combination of 
	# PRODUCE_VOCAB, NLP_PARAMS, and MODELS
	for n, p in zip(neg_kf, pos_kf):
		print 'New Fold'
		
		neg_train = n[0]
		neg_train_label = neg_label.values[neg_train]
		neg_test = n[1]
		neg_test_label = neg_label.values[neg_test]

		pos_train = p[0]
		pos_train_label = pos_label.values[pos_train]
		pos_test = p[1]
		pos_test_label = pos_label.values[pos_test]

		train_content = pd.DataFrame(np.concatenate([neg['Content'].values[neg_train], pos['Content'].values[pos_train]]), columns=['Content'])['Content']
		train_label   = np.concatenate([neg_train_label, pos_train_label]).ravel() 
		test_content = pd.DataFrame(np.concatenate([neg['Content'].values[neg_test], pos['Content'].values[pos_test]]), columns=['Content'])['Content']
		test_label   = np.concatenate([neg_test_label, pos_test_label]).ravel()

		vocab_options = itertools.product(*PRODUCE_VOCAB.values())
		nlp_options = itertools.product(*NLP_PARAMS.values())

		for vocab_option in vocab_options:
			voc_opt = { k:v for k, v in zip(PRODUCE_VOCAB.keys(), vocab_option) }

			for nlp_option in nlp_options: 
				nlp_opt = { k:v for k, v in zip(NLP_PARAMS.keys(), nlp_option) } 
				voc_opt['nlp_params'] = nlp_opt 
				voc_opt['in_neg_df'] = pd.DataFrame(neg['Content'].values[neg_train], columns=['Content'])['Content']
				voc_opt['in_pos_df'] = pd.DataFrame(pos['Content'].values[pos_train], columns=['Content'])['Content']

				temp, nlp_preprocessor = sa.calculate_ratios(**voc_opt)
				print temp 


				for model, param_dict in MODELS.iteritems():
					combos = itertools.product(*param_dict.values())

					for combo in combos:
						pdict = { k:v for k, v in zip(param_dict.keys(), combo) } 
						clf = model(**pdict)

						train_transformed = nlp_preprocessor.fit_transform(train_content)
						test_transformed  = nlp_preprocessor.transform(test_content)

						clf.fit(train_transformed, train_label)

						this_score = scoring_func(test_label, clf.predict(test_transformed))
						this_to_str = str(vocab_option) + str(nlp_option) + str(combos)
						print this_to_str + ': ' + str(this_score)

						if this_to_str not in result.keys():
							result[this_to_str] = [this_score] 
						else:
							result[this_to_str].append(this_score)

	return result 


if __name__ == '__main__':
	result = main() 
	cPickle.dump(result, open('data/grid_search_result.pkl', 'w'))

















