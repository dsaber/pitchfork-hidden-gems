### Pitchfork's Hidden Gems

LIVE @ [p4k-hg.herokuapp.com](https://www.p4k-hg.herokuapp.com)


#### Description

I created Pitchfork's Hidden Gems as my capstone project at Zipfian Academy. It combines a number of passions - music, data science, data products, and natural language processing - into a single platform for discovering new music.

The premise is simple. People are really good at explaining why they like or dislike something, but they are not so good at mapping those feelings to a numeric rating. By developing an algorithm to score reviews based solely on their probability of being 'Strongly Positive', I hope to turn up interesting albums that music lovers might have missed on the basis of a low review score (and that traditional recommender systems would have a hard time finding).

The data comes from the music editorial site Pitchfork.com. I chose Pitchfork for a number of reasons, including: (A) Familiarity -- I'm an avid reader of the site, and domain knowledge is always helpful; (B) Utility -- Most Pitchfork reviews fall within the scores of 5.5 and 8.0, and it can be challenging to decide what those nebulous middle ground reviews actually mean without reading each and every one of them (and that's a lot of work!); and (C) Selfishness -- I was inspired by all the great 6.0- and 7.0-rated albums I've found through Pitchfork, and I wanted to come up with a programmatic way to turn up more. A music lover is never satisfied.

However, I'm not completely selfish! My code is intended to be general enough so that you could plug in any .csv file and develop a similar model/web application. The only requirement is that you would need 'Score', 'Content', 'Artist' and 'Album' sections. (Or the analogous columns for books, movies, and laundromats).


#### How It Works

Whereas recommender systems use the wisdom of crowds to make recommendations, Pitchfork's Hidden Gems is very much an 'Expert Recommender'. It comes up with a probability that a writer's review of a given album is 'Strongly Positive' and makes recommendations on the basis of this probability. In order to produce a metric that is more intuitively satisifying for its consumer-facing frontend, Pitchfork's Hidden Gems maps the probability to a review score based on the actual distribution of Pitchfork review scores.

I featurized my text data by converting it into a TF-IDF Matrix using 1-, 2-, and 3-grams, and then using this feature set to train a Logistic Regression model. I toyed with the idea of using part of speech tagging, but found that there was no performance boost (in this case, 'performance' is defined by ROC-AUC). I arrived at this approach via an extensive grid search, the details of which you may find in [model/search.py](./model/search.py). 

As far as other preprocessing steps go, I tried to strike a middle ground between keeping my approach broadly generalizable while also removing some noticeable irregularities (whether or not those irregularities actually impacted performance). For example, I noticed that certain artists who had been historically well-reviewed, but who had since seen their careers take a step back, turned up as 'underrated' (e.g., Eminem). To sidestep this issue, when constructing the vocabulary for my TF-IDF matrix, I removed artists ONLY from their associated reviews (i.e., 'Eminem' won't show up in reviews of 'Eminem' albums, but his name will show up in the reviews of other hip hop artists). Analogously, you might not want to recommend every Judd Apatow movie in existence simply because people loved Superbad. (Then again, maybe you do... This is definitely the 'Art' part of Data Science).


#### Performance

As mentioned, the model is trained only on 'Strongly Negative' and 'Strongly Positive' reviews, and when cross-validating this model, we see very strong performance: .88 ROC-AUC. 

Of course, validating recommendations is much more difficult, especially since we're primarily interested in using our model to recommend 'mid-range' albums. I've included some iPython markdown files that step through some of my additional validation steps. You may find these in [model/ipython_markdown/](./model/ipython_markdown).


##### Thanks for Reading! 
###### - Dan  

