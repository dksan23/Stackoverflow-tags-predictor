# -*- coding: utf-8 -*-
"""
Created on Sat Feb  2 19:20:08 2019

@author: HeavyD
"""

import sys
sys.path.append("..")
import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords
from ast import literal_eval
import pandas as pd
import numpy as np
def readata(filename):
    data=pd.read_csv(filename,sep='\t')
    data['tags']=data['tags'].apply(literal_eval)
    return data
train=readata('data/Train.csv')
test=readata('data/Test.csv')

xtrain,ytrain=train['title'].values,train['tags'].values
xtest,ytest=test['title'].values,test['tags'].values

import re
REPLACE_BY_SPACE_RE = re.compile('[/(){}\[\]\|@,;]')
BAD_SYMBOLS_RE = re.compile('[^0-9a-z #+_]')
STOPWORDS = set(stopwords.words('english'))

def text_prepare(text):
    """
        text: a string
        
        return: modified initial string
    """
    text =text.lower() # lowercase text
    text =re.sub(REPLACE_BY_SPACE_RE," ",text) # replace REPLACE_BY_SPACE_RE symbols by space in text
    text =re.sub(BAD_SYMBOLS_RE,"",text) # delete symbols which are in BAD_SYMBOLS_RE from text
    text1=text.split()# delete stopwords from text
    text=" ".join(word for word in text1 if word not in (STOPWORDS))
    return text
xtrain=[text_prepare(text) for text in xtrain]

xtest=[text_prepare(text) for text in xtest]
from sklearn.feature_extraction.text import TfidfVectorizer
vectorizer = TfidfVectorizer(min_df=5, max_df=0.9, ngram_range=(1, 2),token_pattern='(\S+)')
xtrain=vectorizer.fit_transform(xtrain)

xtest=vectorizer.transform(xtest)
from collections import defaultdict
tagcounts=defaultdict(int)

for tags in ytrain:
    for tag in tags:
        tagcounts[tag]+=1
from sklearn.preprocessing import MultiLabelBinarizer
mlv=MultiLabelBinarizer(classes=(sorted(tagcounts.keys())))
ytrain=mlv.fit_transform(ytrain)
ytest=mlv.fit_transform(ytest)   
from sklearn.multiclass import OneVsRestClassifier
from sklearn.svm import LinearSVC
def classifer(xtrain,ytrain):
    svm=LinearSVC()
    orc=OneVsRestClassifier(svm,n_jobs=-1)
    return orc.fit(xtrain,ytrain)
classified=classifer(xtrain,ytrain)
prediction=classified.predict(xtrain) 
predictionscore=classified.decision_function(xtrain)   
prediction=mlv.inverse_transform(prediction)
yt=mlv.inverse_transform(ytest)
yt[3]
prediction[3]

from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score
from sklearn.metrics import roc_auc_score 
from sklearn.metrics import average_precision_score
from sklearn.metrics import recall_score
print(accuracy_score(yt, prediction))
print(f1_score(yt, prediction, average='weighted'))
print(average_precision_score(yt, prediction))  

