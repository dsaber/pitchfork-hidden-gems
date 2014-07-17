# don't write bytecodes!
import sys
sys.dont_write_bytecode = True

# web things
from flask import Flask
from flask import render_template
from flask import request

# data/database
import psycopg2
import pandas as pd
import cPickle

# predict logic
import model.predict as mp


app = Flask(__name__)

@app.route('/')
def home_page():
	return '<h1>Hello!</h1>'

# recommender logic



# predicting and scoring logic
@app.route('/predict/')
def predict():
	return render_template('predict.html')

@app.route('/predict/score', methods=[ 'POST' ])
def score():
	text = request.form['review_text']
	log_prob, imputed_score = mp.predict_one(text, tfidf, logreg, scoring_scale, p4k_scoring_map)
	return str(imputed_score) 



if __name__ == '__main__':

	conn = psycopg2.connect(dbname='pitchfork', user='postgres', host='/tmp/')
	cur = conn.cursor()

	tfidf, logreg, p4k_scoring_map, scoring_scale = cPickle.load(open('tfidf_logreg_maps.pkl', 'r'))

	app.run(debug=True)

