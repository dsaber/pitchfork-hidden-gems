# for mapping logistic regression probability to score
import map_scale as ms

def predict_one(review_text, tfidf, logreg, scale, scoring_map):

	transformed_review_text = tfidf.transform([review_text])
	raw_log_prob = logreg.predict_proba(transformed_review_text)[:, 1][0]
	return raw_log_prob, ms.map_to_scoring_map(raw_log_prob, scale, scoring_map)