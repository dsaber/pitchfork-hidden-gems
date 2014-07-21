# don't write bytecodes!
import sys
sys.dont_write_bytecode = True

# web things
from flask import Flask
from flask import render_template
from flask import request
from flask import escape

import urllib2

# data/database
import psycopg2
import pandas as pd
import cPickle

# predict logic
import model.predict as mp

# standard
import numpy as np


app = Flask(__name__)

@app.route('/')
def home_page():
	return render_template('index.html')

# recommender logic
@app.route('/recommend/')
def recommend():
	# pick random albums that have a 'Best New Music' tag
	# so that the user can select one
	to_ex = '''
				SELECT "Album", "Artist", "Artwork", "Link"
				FROM "review" r
				WHERE r."BNM" = 1;
			'''
	cur.execute(to_ex)
	bnm_recs = cur.fetchall()
	np.random.shuffle(bnm_recs)
	random_recs = bnm_recs[:5]

	return render_template('recommend.html', albums=random_recs, ulib=urllib2.quote)

@app.route('/similar/<artist_name>/<page>')
def similar(artist_name, page=0):
	# going to suggest most underrated albums depending on the 
	# genre the user expresses interest in
	get_genre = '''
					SELECT r."Genre"
					FROM "review" r
					WHERE r."Artist" = \'''' + str(artist_name) + "';"
	cur.execute(get_genre)
	genre = cur.fetchone()[0]
	
	get_most_underrated_artists = '''
		SELECT "Album", "Artist", "Artwork", "Link", "Score", "MY_scaled"
		FROM "review" r
		WHERE r."Mid?" = 1 AND r."MY_scaled" - r."Score" > 0 AND
		r."Genre" = \'''' + str(genre) + '''\'
		ORDER BY r."MY_scaled" - r."Score" DESC, r."Score";'''
	cur.execute(get_most_underrated_artists)
	underrated = cur.fetchall()

	begin = 5 * int(page)
	end = begin + 5
	underrated = underrated[begin:end]

	begin_page = str(int(page) - 1)
	end_page   = str(int(page) + 1)

	return render_template('choices.html', underrated=underrated, artist_name=artist_name, begin_page=begin_page, end_page=end_page, ulib=urllib2.quote, int=int)


# predicting and scoring logic
@app.route('/predict/', methods=[ 'GET', 'POST' ])
def predict():
	try: 
		text = request.form['review_text']
		log_prob, imputed_score = mp.predict_one(text, tfidf, logreg, scoring_scale, p4k_scoring_map)
	except:
		imputed_score = ''

	return render_template('predict.html', score=imputed_score)



if __name__ == '__main__':

	conn = psycopg2.connect(dbname='d3hohnj2pptghg', user='miypbfxztnggxg', host='ec2-54-197-241-64.compute-1.amazonaws.com', password='-kOI4Ikgy4D_8YN2N-Bs2nw5jg')
	cur = conn.cursor()

	tfidf, logreg, p4k_scoring_map, scoring_scale = cPickle.load(open('tfidf_logreg_maps.pkl', 'r'))

	app.run(host='0.0.0.0')

