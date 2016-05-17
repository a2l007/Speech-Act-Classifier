from __future__ import division
import inspect
import random
from swa import Transcript
from os import listdir
from time import time
from re import findall, sub
from sklearn.multiclass import OneVsOneClassifier, OneVsRestClassifier
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from collections import Counter
from nltk import word_tokenize
from nltk.corpus import stopwords

__author__ = 'Chitesh Tewani, Atul Mohan, Prashanth Balasubramani'


class Generator:
    def __init__(self, pathToFile):
        self.data = []
        self.trainData = []
        self.pathToFile = pathToFile
        self.testData = []
        self.swbdtrainPercentage = 3
        self.mrdatrainPercentage = 2
        self.swbdtestPercentage = 20
        self.mrdatestPercentage = 20
        self.isswbdTop = 0
        self.isShuffleNeeded = 1
        self.isswbdTest = 1
        self.swbdfile = pathToFile + "SwitchBoard_Merged_Data.csv"
        self.mrdafile = pathToFile + "MRDA_Aggregated.out"
        self.testfile = pathToFile + "Merged_Testset.csv"
        self.trainfile = pathToFile + "Merged_Trainset.csv"

    def mergeData(self):
        swbd = open(self.swbdfile, 'r');
        mrda = open(self.mrdafile, 'r');
        swbdlines = swbd.readlines();
        mrdalines = mrda.readlines();
        swbdtrain = swbdlines[:int(self.swbdtrainPercentage / 100 * len(swbdlines))]
        mrdatrain = mrdalines[:int(self.mrdatrainPercentage / 100 * len(mrdalines))]
        if self.isswbdTop:
            self.trainData.extend(swbdtrain)
            self.trainData.extend(mrdatrain)
        else:
            self.trainData.extend(mrdatrain)
            self.trainData.extend(swbdtrain)
        swbdtest = swbdlines[int((100 - self.swbdtestPercentage) / 100 * len(swbdlines)):]
        mrdatest = mrdalines[int((100 - self.mrdatestPercentage) / 100 * len(mrdalines)):]
        if self.isswbdTest == 1:
            self.testData.extend(swbdtest)
        # self.testData.extend(mrdatest)
        else:
            self.testData.extend(mrdatest)
        # self.testData.extend(swbdtest)
        if self.isShuffleNeeded:
            random.shuffle(self.trainData)
            random.shuffle(self.testData)
        swbd.close()
        mrda.close()

    def writeData(self):
        trainF = open(self.trainfile, 'a');
        testF = open(self.testfile, 'a');
        for w in self.trainData:
            trainF.write(w)

        for wr in self.testData:
            testF.write(wr)
        trainF.close()
        testF.close()


def main():
    gen = Generator('../Data/Merged/')
    dataStartTime = time()
    gen.mergeData()
    gen.writeData()
    dataEndTime = time()
    print "Data loaded in", dataEndTime - dataStartTime, "sec"


if __name__ == '__main__':
    main()
