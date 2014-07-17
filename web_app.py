# don't write bytecodes!
import sys
sys.dont_write_bytecode = True

# web things
from flask import Flask, render_template

# data/database
import psycopg2
import pandas as pd
import cPickle

# predict logic
import model.predict as predict


app = Flask(__name__)

@app.route('/')
def home_page():
	return '<h1>Hello!</h1>'

# predicting and scoring logic
@app.route('/predict/')
def predict():
	return render_template('predict.html')

@app.route('/predict/show_score/<review_text>')
def show_score():
	pass 



if __name__ == '__main__':

	conn = psycopg2.connect(dbname='pitchfork', user='postgres', host='/tmp/')
	cur = conn.cursor()

	tfidf, logreg, p4k_scoring_map, scoring_scale = cPickle.load(open('tfidf_logreg_maps.pkl', 'r'))

	app.run(debug=True)

