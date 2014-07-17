# don't write bytecodes!
import sys
sys.dont_write_bytecode = True

# web things
from flask import Flask

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


if __name__ == '__main__':

	conn = psycopg2.connect(dbname='pitchfork', user='postgres', host='/tmp/')
	cur = conn.cursor()

	tfidf, logreg, p4k_scoring_map, scoring_scale = cPickle.load(open('tfidf_logreg_maps.pkl', 'r'))

	app.run(debug=True)

