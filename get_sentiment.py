# interact with NLTK Sentiment Analysis API
import requests
from _variables import * 

# standard 
import pandas as pd
import numpy as np 


# use this function to interact with an external API that produces probability of 
# negative/positive sentiment
def get_nltk_sentiment(review_content, api_url='https://japerk-text-processing.p.mashape.com/sentiment/', key=MASHAPE_KEY):

	text = { 'text': review_content.strip() }
	headers = { 'X-Mashape-Key': key } 
	r = requests.post(api_url, headers=headers, data=text) 
	try: 
		sentiment = r.json()
		return sentiment['label'], sentiment['probability']['neg'], sentiment['probability']['neutral'], sentiment['probability']['pos']
	except:
		return 'NA', np.nan, np.nan, np.nan 

if __name__ == '__main__':

	df = pd.read_csv('data/p4k_complete_data.csv')
	df['NLTK_label'] = ''
	df['NLTK_pos'] = 0.0
	df['NLTK_neutral'] = 0.0
	df['NLTK_neg'] = 0.0

	for i in xrange(df.shape[0]):
		print i 
		label, neg, neutral, pos = get_nltk_sentiment(df['Content'][i])
		print str(df['Score'][i]) + ' Score against ' + str(10.0 * pos) + ' Sentiment'

		df['NLTK_label'][i] = label
		df['NLTK_pos'][i] = pos
		df['NLTK_neutral'][i] = neutral
		df['NLTK_neg'][i] = neg

	df.to_csv('data/final_data.csv')
	df.to_json('data/final_data.json') 
