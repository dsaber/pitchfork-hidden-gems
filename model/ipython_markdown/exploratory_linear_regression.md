
In[1]:

```
# standard
%matplotlib inline
import pandas as pd
import numpy as np
import nlp_processing as nlpp

# storage
import cPickle

# Linear Regression + friends
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn import metrics
import sklearn.cross_validation as cross_validation
```

In[2]:

```
tfidf, logreg = cPickle.load(open('../tfidf_logreg.pkl', 'r'))
```

In[3]:

```
df = pd.read_csv('../data/final_p4k.csv')
```

In[4]:

```
label = df['Score']
```

In[5]:

```
# Here, I am essentially validating a tangential premise of my project -- 
# specifically, people are really good at explaining whether or not they like something
# and why, but are terrible at assigning numeric values to that feeling (yes, even 
# people who are paid to assign numeric values to their feelings). 

# As we can see, when building a linear regression against review content, we see 
# that review content really only accounts for a relatively small proportion of the overall score.
# Contrast this with the validation we performed in 'search.py,' wherein we saw 
# that our logistic regression model was highly successful (.88 ROC-AUC/87% Accuracy)
# in assigning "Positive"/"Negative" probabilities to reviews

for train, test in cross_validation.KFold(df.shape[0], n_folds=3, shuffle=True):
    print '*****New Fold*****'
    train_content = df['Content'][train]
    test_content  = df['Content'][test]
    train_label   = label[train]
    test_label    = label[test]
    
    svd = TruncatedSVD(250)
    tfidf = TfidfVectorizer(ngram_range=(1, 3), min_df=9)
    transformed_train = svd.fit_transform(tfidf.fit_transform(train_content))
    transformed_test = svd.transform(tfidf.transform(test_content))
    
    clf = RandomForestRegressor(50, n_jobs=-1)
    clf.fit(transformed_train, train_label.ravel())
    
    print '\nRandom Forest Regressor:'
    print 'Train: ' + str(metrics.r2_score(train_label, clf.predict(transformed_train)))
    print 'Test: ' +  str(metrics.r2_score(test_label, clf.predict(transformed_test)))
    print 'Score: ' + str(clf.score(transformed_test, test_label))
    
    clf = LinearRegression()
    clf.fit(transformed_train, train_label.ravel())
    
    print '\nLinear Regression:'
    print 'Train: ' + str(metrics.r2_score(train_label, clf.predict(transformed_train)))
    print 'Test: ' +  str(metrics.r2_score(test_label, clf.predict(transformed_test)))
    print 'Score: ' + str(clf.score(transformed_test, test_label)) + '\n\n'    
```


    *****New Fold*****
    
    Random Forest Regressor:

