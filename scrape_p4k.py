# python libraries useful for scraping
import requests
import bs4 

# storing and interacting with data
import pandas as pd


def scrape_meta(num_pages=780):

	'''This function scrapes all-review related meta data 
	on Pitchfork.com. This meta data includes the following fields:
		- Artist
		- Album
		- Reviewer
		- Score
		- Album Art URL
	'''


	# to be converted to DataFrame
	result = { 'Artist': 		[],
			   'Album':  		[],
			   'Reviewer': 		[],
			   'Date':			[], 
			   'Score': 		[],
			   'Artwork':		[]
			 }


	# Pitchfork's review section
	url_root = 'http://pitchfork.com/reviews/albums/'

	for page in range(num_pages):
		url = url_root + str(page)
		r = requests.get(url)
		bs = bs4.BeautifulSoup(r.text) 

		review_grid = bs.select('.object-grid')[0]
		meta_data = review_grid.select('.info')

		for album in meta_data:
			result['Artist'] = album.h1.text.encode('ascii', 'ignore')
			result['Album']  = album.h2.text.encode('ascii', 'ignore')
			result['Reviewer'] = album.h3.text.encode('ascii', 'ignore')[3:] # [3:] in order to get rid of "by"
			result['']
		