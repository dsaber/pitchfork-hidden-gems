
In[1]:

```
# standard
%matplotlib inline
import pandas as pd
import numpy as np

# storage
import cPickle

# clustering
from sklearn.cluster import KMeans
```

In[57]:

```
tfidf, logreg, p4k_scoring_map, my_score_scale = cPickle.load(open('../tfidf_logreg_maps.pkl', 'r'))
```

In[60]:

```
df = pd.read_csv('../data/final_p4k.csv')
columns_to_keep = [
    'Album',
    'Artist',
    'Content',
    'Date',
    'Reviewer',
    'Score',
    'Genre',
    'Mid?',
    'NLTK_score',
    'TB_score',
    'MY_score',
    'NLTK_scaled',
    'TB_scaled',
    'MY_scaled'
]
df = df[columns_to_keep]
df = df.dropna()
df.head()
```






In[61]:

```
# we're going to want to cluster by reviwer to avoid 

print df['Reviewer'].value_counts()
```


    Joe Tangari           813
    Stephen M. Deusner    657
    Ian Cohen             553
    Brian Howe            478
    Mark Richardson       430
    Marc Hogan            426
    Stuart Berman         331
    Grayson Currin        323
    Nate Patrin           304
    Dominique Leone       273
    Jess Harvell          272
    Matthew Murphy        269
    Jason Crock           267
    Rob Mitchum           264
    Marc Masters          264
    ...
    PJ Gallagher                                                   1
    Brandon Wall                                                   1
    Brian Howe & Brandon Stosuy                                    1
    Adam Dlugacz                                                   1
    Kim Fing Shannon                                               1
    Erin Macleod                                                   1
    Brock Kappers                                                  1
    Alan Smithee                                                   1
    Choppa Moussaoui, with help from Mullah Omar, Ethan P, and     1
    Bruce Tiffee                                                   1
    Matt Wellins                                                   1
    Andy Beta, Brandon Stosuy & Mark Richardson                    1
    Zach Hammerman                                                 1
    Michael Wartenbe                                               1
    Martin Clark                                                   1
    Length: 314, dtype: int64


In[62]:

```
print df['Genre'].value_counts()
```


    Alternative & Punk,Alternative Rock,Alternative                 1066
    Alternative & Punk,General Indie Rock,Indie Rock                 904
    Alternative & Punk,Indie Pop,Indie Rock                          852
    Rock,General Mainstream Rock,Mainstream Rock                     741
    Electronica,Pop Electronica,Electronica Mainstream               389
    Alternative & Punk,Alternative Folk,Alternative Folk             371
    Alternative & Punk,Alternative Singer-Songwriter,Alternative     347
    Alternative & Punk,Neo-Psychedelic,Indie Rock                    347
    Alternative & Punk,Experimental Rock,Indie Rock                  330
    Electronica,Intelligent (IDM),Techno                             329
    Alternative & Punk,Post-Punk Revival,Indie Rock                  291
    Urban,Underground Hip-Hop/Rap,Western Hip-Hop/Rap                265
    Alternative & Punk,Post-Rock,Indie Rock                          263
    Electronica,Ambient Electronica,Downtempo, Lounge & Ambient      261
    Alternative & Punk,Dream Pop,Indie Rock                          261
    ...
    Traditional,Iranian,Asian Traditional                         1
    Traditional,Klezmer & European Jewish,European Traditional    1
    Classical,Baroque Instrumental,Baroque Era                    1
    Traditional,Caribbean Traditional,Caribbean                   1
    Jazz,Boogie-Woogie Piano,Early Jazz                           1
    Rock,Power Metal,Metal                                        1
    Other,Lullabies,Children's                                    1
    Other,New Age Pop,New Age                                     1
    Classical,Other Classical Instrumental,Other Classical        1
    Other,Electronic Space Music,Meditative & Space               1
    Rock,Turkish Rock,European Rock                               1
    Traditional,Hawaiian,Other Traditions                         1
    Classical,Modern Era Avant Garde,Modern Era                   1
    Other,Andean,South American                                   1
    Traditional,Texas Blues,Electric Blues                        1
    Length: 331, dtype: int64


In[67]:

```
df_sub = df[df['Reviewer'] == 'Joe Tangari']
print df_sub.shape
```


    (813, 14)


In[68]:

```
joe = tfidf.transform(df_sub['Content'])
```

In[53]:

```
km = KMeans(4)
km.fit(joe)
df_sub['Cluster'] = km.predict(joe)
```

In[54]:

```
# There are some very compelling things happening here.
# By way of background, first note that we selected an individual reviewer on which to cluster. This 
# hopefully prevents us from clustering based on reviewer writing style and diction and the like.
# Furthermore, we chose a reviewer - Joe Tangari - who only covers very specific 
# genres of music (world and 'fringe' indie). Again, this should hopefully dissaude any fears that 
# we're clustering om something other than sentiment. 

# Now, consider the difference in actual Pitchfork 'Score' between 'Mid'-range 
# and non-'Mid'-range among all clusters. It's 0.2 for Cluster 0, -0.87 for Cluster 1,
# 0.13 for Cluster 2 and 0.01 for Cluster 3. Next, consider the differences in my score. It's .06
# for Cluster 0, -0.55 for Cluster 1, 0.08 for Cluster 2, and -0.1 for Cluster 3. In other words, my
# difference is smaller in 3 of 4 cases, which gives me hope that I'm capturing a rawer, realer
# form of criticism with my model. (In the worst case, it gives me enough hope to build 
# a prototype for our two-week project).

# What's happening within the clusters, however, is  more interesting. Take Cluster 1, for
# example, in which 'Mid'-range reviews have been scaled up considerably - enough to 
# approach Tangari's scores for non-'Mid'-range reviews. Thus, there are probably some interesting 
# albums to be found in (Cluster, Mid) = (1, 1). In other Clusters, we see the complement 
# happening: 'Mid'-range reviews don't get such a dramatic boost, and come in mostly in 
# line with their Score. Importantly, non-'Mid'-range reviews aren't scaled up or down 
# dramatically either. 

df_inspect = df_sub[['Score', 'NLTK_scaled', 'MY_scaled']].groupby([df_sub['Cluster'], df['Mid?']]).mean()
df_inspect['Counts'] = df_sub['Cluster'].groupby([df_sub['Cluster'], df['Mid?']]).count()
df_inspect
```






In[55]:

```
# For argument's sake, let's consider another reviewer, Grayson Currin, who 
# writes primarily about metal music. Again, we see some
# interesting things (in line with my discussion above).

df_sub = df[df['Reviewer'] == 'Grayson Currin']
grayson = tfidf.transform(df_sub['Content'])
km = KMeans(4)
km.fit(grayson)
df_sub['Cluster'] = km.predict(grayson)
df_inspect = df_sub[['Score', 'NLTK_scaled', 'MY_scaled']].groupby([df_sub['Cluster'], df['Mid?']]).mean()
df_inspect['Counts'] = df_sub['Cluster'].groupby([df_sub['Cluster'], df['Mid?']]).count()
df_inspect
```





