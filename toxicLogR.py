#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Wed Dec 20 09:20:21 2017

@author: Taranpreet singh
    """

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.multiclass import OneVsRestClassifier
from sklearn.pipeline import Pipeline
#from sklearn.ensemble import RandomForestClassifier


train = pd.read_csv('../input/train.csv')


#train.info()

#train.describe()
#train.comment_text.head()

#creating x and y
x=train.loc[:,'comment_text']

y = train.drop(['id','comment_text'],axis=1)

#tokens on alphanumeric
tks = '[A-Za-z0-9]+(?=\\s+)'



# creating pipe line to fit 
#Pipelines help a lot when trying different cominations
pl = Pipeline([
        ('vec', CountVectorizer(token_pattern = tks)),
        ('clf', OneVsRestClassifier(LogisticRegression()))
    ])

# Fit to the training data
pl.fit(x,y)

test = pd.read_csv('../input/test.csv')
test.info()
#1 missing value


test = test.fillna("")
#predicting
predictions = pl.predict_proba(test.comment_text)

# Format predictions in DataFrame: prediction_df
prediction_df = pd.DataFrame(columns=y.columns,
                             index=test.id,
                             data=predictions)


# Save prediction_df to csv
prediction_df.to_csv('predictions.csv')
