# suppress .pyc
import sys 
sys.dont_write_bytecode = True 

# python libraries useful for scraping
import requests
import bs4 

# storing and interacting with data
import pandas as pd
import cPickle


def scrape_pitchfork(url_root='http://pitchfork.com/reviews/albums/', num_pages=780):

	'''This function scrapes all-review data from Pitchfork.com; fields
	include the following:
		- Artist
		- Album
		- Reviewer
		- Date 
		- Link to Review Content
		- Album Art URL
		- Review Content 
		- Score 
		- 'Best New Music' Designation
	'''


	result = { 'Artist': 		[],
			   'Album':  		[],
			   'Reviewer': 		[],
			   'Date':			[], 
			   'Link':			[],
			   'Artwork':		[],
			   'Content':		[],
			   'Score':			[],
			   'BNM':			[] 
			 }


	for page in range(1, num_pages + 1):
		print 'Working on page ' + str(page) + ' out of ' + str(num_pages)

		url = url_root + str(page)
		r = requests.get(url)
		bs = bs4.BeautifulSoup(r.text) 


		# primary container of Pitchfork reviews
		obgrid = bs.select('.object-grid')[0]

		# using said primary container to get the fields we need
		rev_info = obgrid.select('.info')
		rev_links = obgrid.findAll('a')

		for album, link in zip(rev_info, rev_links):


			# the Artist/Album/Reviewer/Date fields should be self-explanatory
			result['Artist'].append(album.h1.text.encode('ascii', 'ignore'))
			result['Album'].append(album.h2.text.encode('ascii', 'ignore'))
			result['Reviewer'].append(album.h3.text.encode('ascii', 'ignore')[3:]) # [3:] in order to get rid of "by"
			result['Date'].append(album.h4.text.encode('ascii', 'ignore'))


			# Links to actual review content
			linkage = link['href']
			result['Link'].append(linkage)


			# get actual review content, along with score and whether or not the 
			# album received the 'Best New Music' designation
			r_link = 'http://pitchfork.com' + linkage
			request_content = requests.get(r_link)
			rev_bs = bs4.BeautifulSoup(request_content.text)


			# main content
			rev_content = rev_bs.select('#main')[0]


			# Artwork, Score, and 'Best New Music'
			try: 
				result['Artwork'].append(rev_content.img['src'])
			except:
				result['Artwork'].append('')
			result['Score'].append(float(rev_content.select('.score')[0].text.encode('ascii', 'ignore')))
			bnm = 1 if 'Best New Music' in rev_content.text else 0 
			result['BNM'].append(bnm)


			# Review Content
			result['Content'].append(rev_content.select('.editorial')[0].text.encode('ascii', 'ignore'))


	return result

		

if __name__ == '__main__':

	result = scrape_pitchfork()

	# pickle result 
	cPickle.dump(result, open('data/raw_data_dump.pkl', 'w'))

	# convert to DataFrame and store as json and as csv
	df = pd.DataFrame(result)
	df.to_csv('data/p4k_data.csv')
	df.to_json('data/p4k_data.json')

