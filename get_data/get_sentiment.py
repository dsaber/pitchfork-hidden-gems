# suppress .pyc
import sys

# interact with NLTK Sentiment Analysis API
import requests
from _variables import *

# TextBlob
from textblob import TextBlob

# standard
import pandas as pd

sys.dont_write_bytecode = True
API_URL = 'https://japerk-text-processing.p.mashape.com/sentiment/'


# use this function to interact with an external API
# that produces probability of negative/positive sentiment
def get_nltk_sentiment(review_content,
                       api_url=API_URL,
                       key=MASHAPE_KEY):
    text = {'text': review_content.strip()}
    headers = {'X-Mashape-Key': key}
    r = requests.post(api_url, headers=headers, data=text)
    try:
        sentiment = r.json()
        return (sentiment['label'],
                sentiment['probability']['neg'],
                sentiment['probability']['neutral'],
                sentiment['probability']['pos'])
    except:
        return ('NA', np.nan, np.nan, np.nan)


def get_text_blob_sentiment(review_content):
    review = TextBlob(review_content)
    return review.sentiment.polarity


if __name__ == '__main__':
    df = pd.read_csv('data/final_p4k.csv')
    df['NLTK_label'] = ''
    df['NLTK_pos'] = 0.0
    df['NLTK_neutral'] = 0.0
    df['NLTK_neg'] = 0.0

    df['TB_pos'] = 0.0

    for i in xrange(df.shape[0]):
        print i
        label, neg, neutral, pos = get_nltk_sentiment(df['Content'][i])
        print str(df['Score'][i]) + ' Score against ' + \
            str(10.0 * pos) + ' Sentiment'

        df['NLTK_label'][i] = label
        df['NLTK_pos'][i] = pos
        df['NLTK_neutral'][i] = neutral
        df['NLTK_neg'][i] = neg
        df['TB_pos'][i] = get_text_blob_sentiment(df['Content'][i])

    df.to_csv('data/final_p4k2.csv')
    df.to_json('data/final_p4k2.json')
