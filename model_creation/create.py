# construct final model (see: Notes) and pickle it;
# additionally, make a new data file for new "midrange"
# reviews (i.e., the things we're going to be interested
# in recommending to users)

# suppress .pyc
import sys 
sys.dont_write_bytecode = True 

# model building materials
import nlp_processing as nlpp
from sklearn.linear_model import LogisticRegression

# standard/storing
import pandas as pd
import cPickle

if __name__ == '__main__':

	pass