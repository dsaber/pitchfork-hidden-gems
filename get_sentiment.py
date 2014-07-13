# interact with NLTK Sentiment Analysis API
import requests

# standard 
import pandas as pd


# use this function to interact with an external API that produces probability of 
# negative/positive sentiment
def get_nltk_sentiment(review_content, api_url='http://text-processing.com/api/sentiment/'):

	payload = { 'text': review_content.strip() }
	r = requests.post(api_url, data=payload) 

	if r.status_code == 503:
		print 'Upgrade to Premium'
		return -1 
	else: 
		sentiment = r.json()
		return sentiment['label'], sentiment['probability']['neg'], sentiment['probability']['neutral'], sentiment['probability']['pos']


if __name__ == '__main__':

	df = pd.read_csv('data/p4k_complete_data.csv')
	df['NLTK_label'] = ''
	df['NLTK_pos'] = 0.0
	df['NLTK_neutral'] = 0.0
	df['NLTK_neg'] = 0.0

	for i in xrange(df.shape[0]):
		label, neg, neutral, pos = get_nltk_sentiment(df['Content'][i])

		df['NLTK_label'][i] = label 
		df['NLTK_pos'][i] = pos
		df['NLTK_neutral'][i] = neutral
		df['NLTK_neg'][i] = neg

	df.to_csv('data/final_data.csv')
	df.to_json('data/final_data.json') 
