





def produce_sentiment_data(in_df, neg_thresh=5.0, pos_thresh=8.0, reviewer='All', genre='All'):

	'''This function is used to produce positive and negative data sets (i.e., DataFrames)
	that are going to be used in model-building. The reviewer and genre fields are 
	optional, but depending on the circumstances, can be useful to gain deeper
	understanding into a specific critic's writing/feelings about a particular
	genre'''

	df = in_df.copy()
	if reviewer != 'All':
		df = df[df['Reviewer'] == reviewer]
	if genre != 'All':
		df = df[df['Genre'] == genre]

	neg = df['Score'] <= neg_thresh
	pos = df['Score'] >= pos_thresh
	to_classify = 

	return df[~(neg | pos)], df[neg], df[pos] 


