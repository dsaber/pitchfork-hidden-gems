# This .py file isn't a module so much as it is a workhorse:
# Its primary goal is to find the best version of the following data pipeline:
	# (A) CV and TF-IDF (i.e., tuning minimum document frequency, ngram range, etc.)
	# (B) "Importance Threshold" (i.e., to be included as a feature, a word or ngram
	# 	   must show up in far more 'positive' documents than 'negative' documents, 
	#	   or vice versa. The "far more" is the threshold of interest.
	# (C) Naive Bayes OR Logistic Regression (including tuning parameters)

# suppress .pyc
import sys 
sys.dont_write_bytecode = True 

# my sentiment analysis tools
import nlp_processing as nlpp

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
							'cv_or_tfidf': 		[ 'TFIDF' ],
							'info_thresh': 		[ None ]
}
NLP_PARAMS = 		  { 
							'tokenizer': 		[ nlpp.POSTokenizer() ],
							'ngram_range': 		[ (1, 2) ], 
							'min_df': 			[ 9 ]
}
NAMES = 			  { 
							LogisticRegression: 'LogisticRegression'
}
MODELS = 			  { 
							LogisticRegression: { 'penalty': 		[ 'l2' ],
								  						'C':		[ 100.0, 500.0 ] }
}


# This is essentially a gigantic grid search
def main(scoring_func=metrics.roc_auc_score, file_path='data/final_p4k.csv', 
		 vocab_dict=PRODUCE_VOCAB, nlp_dict=NLP_PARAMS, model_names=NAMES, models=MODELS):

	# our result is going to be a dictionary where keys correspond to 
	# combinations of NLP features crossed with combinations of Algorithms/Associated 
	# Tuning Parameters
	result = {} 

	df = pd.read_csv(file_path)
	mid, neg, pos = nlpp.produce_sentiment_data(df)

	# create labels
	neg_label = pd.DataFrame([0] * neg.shape[0])
	pos_label = pd.DataFrame([1] * pos.shape[0]) 

	# prepare for cross validation
	neg_kf = cross_validation.KFold(neg.shape[0], n_folds=3, shuffle=True)
	pos_kf = cross_validation.KFold(pos.shape[0], n_folds=3, shuffle=True)

	# create Vocab/NLP combinations; NOTE: I needed to convert them to 
	# lists because lazy evaluation causes some strange behavior
	vocab_options = list(itertools.product(*vocab_dict.values()))
	nlp_options   = list(itertools.product(*nlp_dict.values()))

	# loop through train/test folds for any given combination of 
	# vocab_dict, nlp_dict, and models
	for n, p in zip(neg_kf, pos_kf):
		print 'New Fold'
		
		# create training and test sets for this fold
		neg_train_index 	= n[0]
		neg_test_index 		= n[1]

		neg_train 			= pd.DataFrame(neg['Content'].values[neg_train_index], columns=['Content'])['Content']
		neg_test 			= pd.DataFrame(neg['Content'].values[neg_test_index], columns=['Content'])['Content']
		neg_train_label 	= neg_label.values[neg_train_index]
		neg_test_label 		= neg_label.values[neg_test_index]

		pos_train_index 	= p[0]
		pos_test_index 		= p[1]

		pos_train 			= pd.DataFrame(pos['Content'].values[pos_train_index], columns=['Content'])['Content']
		pos_test 			= pd.DataFrame(pos['Content'].values[pos_test_index], columns=['Content'])['Content']
		pos_train_label 	= pos_label.values[pos_train_index]
		pos_test_label 		= pos_label.values[pos_test_index]

		# consolidate all data for training/testing
		train_content 		= pd.concat([neg_train, pos_train])
		train_label  	    = np.concatenate([neg_train_label, pos_train_label]).ravel() 
		test_content 		= pd.concat([neg_test, pos_test])
		test_label   		= np.concatenate([neg_test_label, pos_test_label]).ravel()


		for vocab_option in vocab_options:
			voc_opt = { k:v for k, v in zip(vocab_dict.keys(), vocab_option) }

			for nlp_option in nlp_options: 
				nlp_opt = { k:v for k, v in zip(nlp_dict.keys(), nlp_option) } 

				# Avoid some computationally expensive processes
				if nlp_opt['ngram_range'] != (1, 1) and (voc_opt['info_thresh'] is not None or nlp_opt['min_df'] == 1):
					pass
				else: 
					voc_opt['nlp_params'] = nlp_opt
					voc_opt['in_neg_df'] = neg_train
					voc_opt['in_pos_df'] = pos_train 

					nlp_preprocessor = nlpp.build_cv_or_tfidf(**voc_opt)

					# modeling process:
					# (1) fit CV/TFIDF on training set; (2) transform training set with 
					# NLP processing tool from (1); (3) fit model on transformed training set 
					# from (2); (4) transform test set using NLP processing tool from (1);
					# (5) compute scoring metric by using fitted model on transformed test data 
					for model, param_dict in models.iteritems():
						combos = itertools.product(*param_dict.values())

						for combo in combos:
							pdict = { k:v for k, v in zip(param_dict.keys(), combo) } 
							clf = model(**pdict)

							# (1)-(2)
							train_transformed = nlp_preprocessor.fit_transform(train_content)
			
							# (3) 
							clf.fit(train_transformed, train_label)

							# (4) 
							test_transformed  = nlp_preprocessor.transform(test_content)

							# (5)
							this_score = scoring_func(test_label, clf.predict(test_transformed))

							# Keep track of information in result
							this_to_str = str(vocab_option) + str(nlp_option) + model_names[model] + str(combo)
							if this_to_str not in result.keys():
								result[this_to_str] = [this_score] 
							else:
								result[this_to_str].append(this_score)

							print this_to_str + ': ' + str(this_score)

	return result 


if __name__ == '__main__':
	result = main() 
	cPickle.dump(result, open('model/grid_search/grid_search_complete.pkl', 'w'))

