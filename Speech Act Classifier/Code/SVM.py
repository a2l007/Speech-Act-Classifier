#!/usr/bin/env python
# -*- coding: utf-8 -*-
#######################
__version__ = "0.2"
__date__ = "Nov. 16, 2015"
__author__ = "Muhammad Abdul-Mageed"
"""
This code is distributed in SMM Z639.
Although this code runs, it may have bugs and is sub-optimal.
You can use the code, change it., etc. but please do not re-distribute.
Let me know if you catch any bugs.
"""
####################################
import argparse
import codecs
import time
import sys
import os, re
import nltk
from collections import defaultdict
from random import shuffle, randint
import numpy as np
from numpy import array, arange, zeros, hstack, argsort
import unicodedata
from scipy.sparse import csr_matrix
# sklearn imports
from sklearn.svm import SVC
from sklearn import preprocessing
from sklearn.cross_validation import StratifiedKFold
from sklearn.grid_search import GridSearchCV
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.cross_validation import train_test_split
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_selection import SelectKBest, f_classif, chi2
from sklearn.multiclass import OneVsOneClassifier, OneVsRestClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
n_jobs = 25

def getListOfLines():
    """
    Just takes a file and returns a list of its line
    """
    #hard-coding path to inputfile for now
    infileObject=codecs.open("pathToFile", "r", "utf-8")
    listOfLines= infileObject.readlines() 
    return listOfLines

#
#####################################
def getSAIMAThreeColumnFormat():
    """
    """
    infileObject=codecs.open("pathToFile", "r", "utf-8")
    listOfLines= infileObject.readlines() 
    saimaDataTuples=[(line.split("\t")[1], line.split("\t")[2].lower()) for line in listOfLines if line.split("\t")[1] !="NO-EMOTION"]
    return saimaDataTuples
    
def tagInSecondHalf(tag, tweet):
    """
    Conditioning position of tag in tweet.
    P.S. Won't consider a tag like #happyday.
    """
    tags= ["#happy", "#sad", "#disgusted", "#fearful" , "#surprised", "#angry"] #"#scared"
    tweet=tweet.split()
    if tag not in tweet:
        return False
    midPoint=(len(tweet)/2)
    tagIndex=tweet.index(tag)
    if tagIndex > midPoint:
        return True
    return False

def tagInLastThird(tag, tweet):
    """
    Conditioning position of tag in tweet.
    P.S. Won't consider a tag like #happyday.
    """
    tweet=tweet.split()
    if tag not in tweet:
        return False
    thirdPoint=(len(tweet)/4)
    tagIndex=tweet.index(tag)
    if tagIndex > thirdPoint*3:
        return True
    return False

def pure(tag, tweet):
    tagList= ["#happy", "#sad", "#disgusted", "#fearful" , "#surprised", "#angry", "#scared"]
    tagList.remove(tag)
    for t in tagList:
        if t in tweet: 
            return False
    return True

def removeSeed(seed, tweet):
    """
    """
    if type(seed)==str:
        tweet= re.sub(seed, " ", tweet)
    elif type(seed)==list:
        for t in seed:
            tweet= re.sub(t, " ", tweet)
    else:
        print type(seed)
        print "arg1/Tag must be a string or list, you provided ", type(tag), "."
        exit()
    #clean
    tweet=re.sub("\s+", " ", tweet)
    #tweet=tweet.trim()
    tweet=tweet.rstrip()
    tweet=tweet.lstrip()
    return tweet

def clean(tweet):
    """
    """
    tweet= re.sub(".", " ", tweet)
    return tweet

def longTweet(tweet):
    """
    """
    if len(tweet.split()) > 10:
        return True
    return False
    
#----------------------------------------------
def getDataDict(emotionLines):
    shuffle(emotionLines)
    #emotionLines=emotionLines[:10000]
    tagLexicon= ["happy", "sad", "disgusted", "fearful" , "surprised", "angry", "scared"] #"#scared"
    tagDict= {"happy": "HAPPINESS", "sad": "SADNESS", "disgusted": "DISGUST", "fearful": "FEAR" , "surprised": "SURPRISE", "angry": "ANGER", "scared": "FEAR"} #"#scared"
    myData={}
    for cat in tagLexicon:
        tag="#"+cat
        myData[tagDict[cat]]=[tweet for tweet in emotionLines if tag in tweet.split() and pure(tag, tweet)
                 and tagInSecondHalf(tag, tweet)  and len(tweet.split()) > 4
                 and removeSeed(tag, tweet) and clean(tweet) and longTweet(tweet)]
    # lump "fearful" with "scared"
#     for k in myData:
#         if k=="fearful":
#             myData["scared"].append(myData[k])

#     myData.pop("fearful", None)
    return myData

def getThreeColumnDataDict(emotionLines):
    shuffle(emotionLines)
    #emotionLines=emotionLines[:10000]
    classes= ["HAPPINESS", "SADNESS", "DISGUST", "FEAR" , "SURPRISE", "ANGER"]
    myData={pair[0]: [] for pair in emotionLines}
    for cat in classes:
        for pair in emotionLines:
            if pair[0]==cat:
                myData[pair[0]].append(pair[1])
    return myData

def getDataStats(myData):
    # Print some stats:
    ##########################
    majorClass=max([len(myData[k]) for k in myData])
    totalCount=sum([len(myData[k]) for k in myData])
    print "Majority class count: ", majorClass
    print "Total data point count: ", totalCount
    print "Majority class % in train data: ", round((majorClass/float(totalCount))*100, 2), "%"
    print "*"*50, "\n"

def getLabeledDataTuples(myData):
    # At this point "myData" is a dict, with each emotion class as a key, and related tweet lines as a list of lines
    ###############################################################
    # The below gets me tweet body only (and filters out rest of each tweet line [e.g., tweetId.])
    # newData will be a list of tuples, each tuple has 0 as an emotion class and 1 as the string/unicode of the tweet body
    dataTuples=[(k, "".join(myData[k][i]).split("\t")[-1]) for k in myData for i in range(len(myData[k]))]
    #shuffle(dataTuples)
    #######################################################################
    # See it: 
    #print "The type of newData[0][0] is a: ", type(newData[0][0]), newData[0][0] # --> newData[0] is a string
    #print "The type of newData[0][1] is a: ", type(newData[0][1]), newData[0][1] # --> newData[1] is a unicode of tweet body
    #######################################################################
    return dataTuples
    
def getFeatures(dataPoint):
    features=defaultdict()
    # label is class name, of course, and feats is just a list of words in this case.
    label, feats=dataPoint[0], dataPoint[1].split()
    # I could also add some code to remove the seeds from the feature dict instead of the heavy computation in
    # the tweet cleaning in removeSeed
    ###########################################
    # Beautify the below, building "has(word): True/False" dict
    for i in feats:
        features[i]=i
    if "#fearful" in features:
        del features["#fearful"]
    if "#scared" in features:
        del features["#scared"]
    return features, label

#featuresets=[getFeatures(i) for i in newData]

def getLabelsAndVectors(dataTuples):
    """ 
    Input:
        dataTuples is a list of tuples
        Each tuple in the list has
                   0=label
                   1= tweet body as unicode/string
    Returns an array of labels and another array for words 
    """
    labels=[]
    vectors=[]
    ids=[]
    c=0
    #unicodedata.normalize('NFKD', title).encode('ascii','ignore')
    for dataPoint in dataTuples:
        ids.append(c)
        c+=1
        label, vector=dataPoint[0], dataPoint[1].split()
        #label, vector=dataPoint[0], unicodedata.normalize('NFKD', dataPoint[1]).encode('ascii','ignore').split()
        labels.append(label)
        vectors.append(vector)
    #labels=array(labels)
    #print labels.shape
    #vectors=array(vectors)
    #print vectors.shape
    return ids, labels, vectors

def getSpace(vectors):
    # get the dictionary of all words in train; we call it the space as it is the space of features for bag of words
    space={}
    for dataPoint in vectors:
        words=dataPoint
        for w in words:
            if w not in space:
                space[w]=len(space)
    return space

def getReducedSpace(vectors, space):
    # get the dictionary of all words in train; we call it the space as it is the space of features for bag of words
    reducedSpace=defaultdict(int)
    for dataPoint in vectors:
        words=dataPoint
        for w in words:
            reducedSpace[w]+=1
    for w in space:
        if reducedSpace[w] < 3:
            del reducedSpace[w]
    reducedSpace={w: reducedSpace[w] for w in reducedSpace}
    return reducedSpace

def getOneHotVectors(ids, labels, vectors, space):
    oneHotVectors={}
    triples=zip(ids, labels, vectors)
    vec = np.zeros((len(space)))
    #for dataPoint in vectors:
    for triple in triples:
        idd, label, dataPoint= triple[0], triple[1], triple[2]
        #for t in xrange(len(space)):
        # populate a one-dimensional array of zeros of shape/length= len(space)
        vec=np.zeros((len(space))) # ; second argument is domensionality of the array, which is 1
        for w in dataPoint:
            try:
                vec[space[w]]=1
            except:
                continue
        oneHotVectors[idd]=(vec, array(label))
    return oneHotVectors

def getOneHotVectorsAndLabels(oneHotVectorsDict):
    vectors= array([oneHotVectorsDict[k][0] for k in oneHotVectorsDict])
    labels= array([oneHotVectorsDict[k][1] for k in oneHotVectorsDict])
    print "labels.shape", labels.shape 
    print "vectors.shape", vectors.shape 
    return vectors, labels
###############################
# try:
#     vectors.shape[0]
# except:
#     vectors=zeros(len(vectors))

# Do grid search
#######################################
def SVM_gridSearch(trainVectors, trainLabels, kernel):
    C_range = 10.0 ** arange(-2, 2)
    gamma_range = 10.0 ** arange(-2, 2)
    param_grid = dict(gamma=gamma_range, C=C_range)
    cv = StratifiedKFold(y=trainLabels, n_folds=2)
    grid = GridSearchCV(SVC(kernel=kernel), param_grid=param_grid, cv=cv, n_jobs=n_jobs) #GridSearchCV(SVC(kernel=kernel, class_weight='auto')
    grid.fit(trainVectors, trainLabels)
    ##################################
    ## Estimated best parameters
    C = grid.best_estimator_.C
    gamma = grid.best_estimator_.gamma
    ##################################
    return C, gamma
#######################################

def getCAndGamma(trainVectors, trainLabels, kernel = 'rbf'):
    C, gamma = SVM_gridSearch(trainVectors, trainLabels, kernel)
    print C
    print gamma
    return C, gamma

def isRetweet(tweet):
    if tweet.lower().split()[0] =="re":
        return True
    return False
def main():
    #######################################
    # Saima Aman emotion blog data
    saimaDataTuples=getSAIMAThreeColumnFormat()
    print "Length of saimaDataTuples is: ",  len(saimaDataTuples)
    shuffle(saimaDataTuples)
    print "saimaDataTuples", saimaDataTuples[0]
    trainTuples=saimaDataTuples[:1000]
    testTuples=saimaDataTuples[1000:]

#     #######################################
    myData=getThreeColumnDataDict(saimaDataTuples)
    totalCount=sum([len(myData[k]) for k in myData])
    print totalCount
#     del trainLines
#     print"*"*50
    getDataStats(myData)
#     dataTuples=getLabeledDataTuples(myData)
#     ####################################
#     # Add first 1000 Saima tuples
#     #dataTuples=dataTuples+saimaDataTuples[:1000]
#     print dataTuples[0]
#     del myData
    ids, labels, vectors= getLabelsAndVectors(trainTuples)
    space=getSpace(vectors)
    reducedSpace=getReducedSpace(vectors, space)
    print "Total # of features in your space is: ", len(space)
    print "Total # of features in your reducedSpace is: ", len(reducedSpace)
    oneHotVectors=getOneHotVectors(ids, labels, vectors, space)
    vectors, labels=getOneHotVectorsAndLabels(oneHotVectors)
    del oneHotVectors
    trainVectors = vectors
    trainLabels = labels
    del vectors
    del labels
    #C, gamma = getCAndGamma(trainVectors, trainLabels, kernel = 'rbf')
    # Train classifier
    #clf = OneVsOneClassifier(SVC(C=C, kernel=kernel, class_weight='auto', gamma=gamma, verbose= True, probability=True))
    clf = OneVsRestClassifier(SVC(C=1, kernel = 'linear', gamma=1, verbose= False, probability=False))
    clf.fit(trainVectors, trainLabels)
    print "\nDone fitting classifier on training data...\n"
    del trainVectors
    del trainLabels
#     saimaDataTuples=getSAIMAThreeColumnFormat()
#     print "Length of saimaDataTuples is: ",  len(saimaDataTuples)
#     shuffle(saimaDataTuples)
#     print "saimaDataTuples", saimaDataTuples[0]
    ids, labels, vectors= getLabelsAndVectors(testTuples)
    oneHotVectors=getOneHotVectors(ids, labels, vectors, space)
    vectors, labels=getOneHotVectorsAndLabels(oneHotVectors)
    del oneHotVectors
    testVectors = vectors
    testLabels = labels
    predicted_testLabels = clf.predict(testVectors)
    print "Done predicting on DEV data...\n"
    print "classification_report:\n", classification_report(testLabels, predicted_testLabels)#, target_names=target_names)
    print "accuracy_score:", round(accuracy_score(testLabels, predicted_testLabels), 2)
    #print "\n confusion_matrix:\n", confusion_matrix(testLabels, predicted_testLabels)
if __name__ == "__main__":
    print "Hello!!"
    main()