# for mapping logistic regression probability to score
import map_scale as ms

# for default scoring (i.e., when we have non-review sized input)
from textblob import TextBlob
import cPickle
default_scale = cPickle.load(open('default_scale.pkl', 'r'))

def predict_one(review_text, tfidf, logreg, scale, scoring_map, default_scale=default_scale):

	# my model was trained on reviews, so it doesn't really work well with tweet-sized
	# input, hence the ensembling
	if len(review_text) > 1000: 
		transformed_review_text = tfidf.transform([review_text])
		raw_log_prob = logreg.predict_proba(transformed_review_text)[:, 1][0]
		return raw_log_prob, ms.map_to_scoring_map((10*raw_log_prob), scale, scoring_map)
	else:
		# defaulting to TextBlob scoring
		raw_tblob_score = TextBlob(review_text).sentiment.polarity
		return raw_tblob_score, ms.scale_score(raw_tblob_score)