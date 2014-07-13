# to interact with Gracenote API
import pygn_module.pygn as pygn 
from _variables import * 

# standard
import pandas as pd 


# use this function to interact with Gracenote's API to bring 
# in genre data
def get_genre_data(c_id, u_id, artist_name, album_name):
	metadata = pygn.search(clientID=c_id, userID=u_id, artist=artist_name, album=album_name)
	genre_data = metadata['genre']

	genre_id = []
	genre_name = [] 
	for key, val in genre_data.iteritems(): 
		genre_id.append(genre_data[key]['ID'])
		genre_name.append(genre_data[key]['TEXT'])

	return ','.join(genre_id), ','.join(genre_name)


if __name__ == '__main__':

	# register with Gracenote API: You would need to provide your own Client ID
	USER_ID = pygn.register(CLIENT_ID)

	# read in data
	df = pd.read_csv('data/p4k_data.csv')
	df['GenreID'] = ''
	df['Genre'] = ''
	num_unavailable = 0

	# use 'for' loop to avoid slamming Gracenote with 10's of 1000's of requests
	# concurrently
	for i in xrange(df.shape[0]):
		print i 
		try: 
			album_genre_id, album_genre_name = get_genre_data(CLIENT_ID, USER_ID, df['Artist'][i], df['Album'][i])
			df['GenreID'][i] = album_genre_id
			df['Genre'][i] = album_genre_name
		except:
			print 'Data unavailable'
			num_unavailable += 1
			df['GenreID'][i] = 'XXXX'
			df['Genre'][i] = 'Other'

	print 'Unable to find ' + str(num_unavailable)
	df.to_csv('data/p4k_complete_data.csv')
	df.to_json('data/p4k_complete_data.json')


