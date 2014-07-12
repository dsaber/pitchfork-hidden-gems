# python libraries useful for scraping
import requests
import bs4 

# storing and interacting with data
import pandas as pd


def scrape_meta(url_root='http://pitchfork.com/reviews/albums/', num_pages=780):

	'''This function scrapes all-review related meta data 
	on Pitchfork.com. This meta data includes the following fields:
		- Artist
		- Album
		- Reviewer
		- Album Art URL
	'''


	result = { 'Artist': 		[],
			   'Album':  		[],
			   'Reviewer': 		[],
			   'Date':			[], 
			   'Link':			[],
			   'Artwork':		[]
			 }


	for page in range(num_pages):

		if page == 0:
			str_page = ""
		else:
			str_page = str(page) 

		url = url_root + str_page
		r = requests.get(url)
		bs = bs4.BeautifulSoup(r.text) 

		# primary container of Pitchfork reviews
		obgrid = bs.select('.object-grid')[0]

		# using said primary container to get the fields we need
		rev_info = obgrid.select('.info')
		rev_links = obgrid.findAll('a')
		art_info = obgrid.select('.artwork')

		for album, link, art in zip(rev_info, rev_links, art_info):


			# the Artist/Album/Reviewer/Date fields should be self-explanatory
			result['Artist'].append(album.h1.text.encode('ascii', 'ignore'))
			result['Album'].append(album.h2.text.encode('ascii', 'ignore'))
			result['Reviewer'].append(album.h3.text.encode('ascii', 'ignore')[3:]) # [3:] in order to get rid of "by"
			result['Date'].append(album.h4.text.encode('ascii', 'ignore'))


			# Links to actual review content
			result['Link'].append(link['href'])


			# the Album Art field is a bit trickier because of Pitchfork's peculiar HTML;
			# I'm simply flagging it so you (as well as Future-Me) will know to be wary
			temp_art = bs4.BeautifulSoup(art.select('.lazy')[0]['data-content'])
			result['Artwork'].append(temp_art.img['src'])

	return result

		