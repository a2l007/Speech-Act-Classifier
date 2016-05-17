from __future__ import division
import inspect
from os import listdir
from re import findall, sub
from nltk import word_tokenize, pos_tag
from time import time,sleep
from sklearn.multiclass import OneVsOneClassifier, OneVsRestClassifier
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from collections import Counter


__author__ = 'Chitesh Tewani, Atul Mohan, Prashanth Balasubramani'

class BagOfWords:
    def __init__(self):
        self.space = Counter()

    def populateSpace(self, data):
        # generate space for bag of words on all the data
        # space => {'unique_word': (unique_index_for_word, <number_of_occurences>)..}
        for utter in data:
            #print utter.text
            for token in utter[3]:
               if token in ['{', '}', '[', ']', '/']:  # ignore literals
                   continue
               if self.space[token] == 0:
                    self.space[token] = [len(self.space), 1]
               else:
                    self.space[token][1] += 1

    def featurize(self, utterances):
        feature_vectors = []
        speec_acts = []
        utter_text = []
        # form feature vector for sentences
        for utter in utterances:
            #print utter
            #utterTokens = word_tokenize(utter.text)
            feature_vector_utter = [0] * len(self.space)
            for utterToken in utter[3]:
                if utterToken in ['{', '}', '[', ']', '/']:  # ignore literals
                   continue
                if self.space[utterToken] != 0:
                    feature_vector_utter[self.space[utterToken][0]] = 1     # get the unique index of the word in space
            speec_acts.append(utter[2])
            utter_text.append(utter[1])
            #print utter[2]
            #sleep(0.1)
            feature_vectors.append(feature_vector_utter)

        return feature_vectors, speec_acts, utter_text

class Feature:
    def __init__(self, utterance):
        self.utterance = utterance
        self.featureHeaders = [
            'question_mark',        # check for presence of question mark
            'wh_question',          # check for presence of wh- question words
            #'i_dont_know',          # check for presence of phrase 'i don't know'
            'no_words',             # check for presence of "No" words
            'yes_words',            # check for presence of "Yes" words
            'do_words',             # check for presence of tense of "do" - did, does
            #'non_verbal',           # check for presence of non-verbal words, < action >
            #'UH_count',             # check for presence of Interjection (UH) Parts of speech in the sentence
            #'CC_count',             # check for presence of co-ordinating conjunction (CC)
            'thanking_words',       # check for presence of words expressing "Thanks"
            'apology_words',        # check for presence of words
            #'sub_utterance_index',  # add sub-utterance index
            #'utterance_index',      # add utterance index
            #'utterance_count'       # add conversation length
            "or_question",          # check for presence of "or" words
        ]

        self.featureKeys = {
            "question_mark" :       '?',
            "wh_question"   :       ['who', 'which', 'where', 'what', 'how'],
            "i_dont_know"   :       ["i don't know"],
            "no_words"      :       ["no", "nah"],
            "yes_words"     :       ["yes", "yeah", "right", "true", "yep"],
            "do_words"      :       ["do", "did", "does"],
            "non_verbal"    :       '^<.*?>',
            "UH_count"      :       '/UH',
            "CC_count"      :       '/CC',
            "thanking_words":       ['thank', 'thanks', 'thank you'],
            "apology_words" :       ['sorry', 'apology'],
            "or_question"   :       ['or']
        }

    def question_mark(self):
        if self.featureKeys[inspect.currentframe().f_code.co_name] in self.utterance[1]:
            return 1
        return 0

    def or_question(self):
        tag_word_count = 0
        for tag_word in self.featureKeys[inspect.currentframe().f_code.co_name]:
            if findall('\\b'+tag_word+'\\b', self.utterance[1]):
                tag_word_count += 1
        return tag_word_count

    def wh_question(self):
        tag_word_count = 0
        for tag_word in self.featureKeys[inspect.currentframe().f_code.co_name]:
            if findall('\\b'+tag_word+'\\b', self.utterance[1]):
                tag_word_count += 1
        return tag_word_count

    def i_dont_know(self):
        tag_word_count = 0
        for tag_word in self.featureKeys[inspect.currentframe().f_code.co_name]:
            if findall('\\b'+tag_word+'\\b', self.utterance[1]):
                tag_word_count += 1
        return tag_word_count

    def no_words(self):
        tag_word_count = 0
        for tag_word in self.featureKeys[inspect.currentframe().f_code.co_name]:
            if findall('\\b'+tag_word+'\\b', self.utterance[1]):
                tag_word_count += 1
        return tag_word_count

    def yes_words(self):
        tag_word_count = 0
        for tag_word in self.featureKeys[inspect.currentframe().f_code.co_name]:
            if findall('\\b'+tag_word+'\\b', self.utterance[1]):
                tag_word_count += 1
        return tag_word_count

    def do_words(self):
        tag_word_count = 0
        for tag_word in self.featureKeys[inspect.currentframe().f_code.co_name]:
            if findall('\\b'+tag_word+'\\b', self.utterance[1]):
                tag_word_count += 1
        return tag_word_count

    def non_verbal(self):
        # search for string <abcde>,
        #  ^ -> start of sentence, non-greedy pattern <.*?>
        return len(findall(self.featureKeys[inspect.currentframe().f_code.co_name], self.utterance[1]))

    def UH_count(self):
        # maybe, check for length of text; if length less than 2 then return true? - Skepticism :-/
        if len(self.utterance.text.split()) < 3 and \
                self.featureKeys[inspect.currentframe().f_code.co_name] in self.utterance.pos:
            return 1
        return 0

    def CC_count(self):
        if len(self.utterance.text.split()) < 3 and \
                self.featureKeys[inspect.currentframe().f_code.co_name] in self.utterance.pos:
            return 1
        return 0

    def thanking_words(self):
        tag_word_count = 0
        for tag_word in self.featureKeys[inspect.currentframe().f_code.co_name]:
            if findall('\\b'+tag_word+'\\b', self.utterance[1]):
                tag_word_count += 1
        return tag_word_count

    def apology_words(self):
        tag_word_count = 0
        for tag_word in self.featureKeys[inspect.currentframe().f_code.co_name]:
            if findall('\\b'+tag_word+'\\b', self.utterance[1]):
                tag_word_count += 1
        return tag_word_count

    def sub_utterance_index(self):
            return self.utterance.subutterance_index

    def utterance_index(self):
            return self.utterance.utterance_index

    def utterance_count(self):
            return self.utterance.utterance_count

class Classifier:
    def __init__(self, dataset, datasetPath):
        self.dataset = dataset
        self.datasetPath = datasetPath
        self.data = []
        self.totalDataCount = 0
        self.misMatchSpeechActCount = 0
        self.misMatchSpeechAct = []
        self.speech_acts_class = [
            #'sd',
            #'b',
            #'sv',
            #'aa',
            'qy',
            #'x',
            #'ny',
            'qw',
            #'nn',
            #'h',
            #'qy^d',
            #'qw^d',
            'fa',
            'ft',
            'qrr',
            'qo',
            'qr',
            'ny',
            'nn',
            's'
        ]
        self.speech_acts_class = self.speechActDictify()
        self.speech_acts_class_count = Counter()
        self.trainPercentage = 3
        self.testPercentage = 30

    def speechActDictify(self):
        speech_acts_class = Counter()
        for speech_act in self.speech_acts_class:
            speech_acts_class[speech_act] = 1

        return speech_acts_class

    def filterSpeechAct(self, speechAct):
            #speechActMod = sub('\^[a-z,0-9]+','', speechAct)
            found_S_tag = False
            speechActMod = speechAct.split('|')
            for act in speechActMod:
                if act == 's':
                    found_S_tag = True
                elif self.speech_acts_class[act] != 0:
                    return act

            speechActMod = speechAct.split('^')
            for act in speechActMod:
                if act == 's':
                    found_S_tag = True
                elif self.speech_acts_class[act] != 0:
                    return act

            '''
            speechActNorm = self.normalizemrdatosw(speechAct)
            if self.speech_acts_class[speechActMod] != 1:
                if len(speechActNorm) != 0 and self.speech_acts_class[speechActNorm] != 0:
                    speechAct = speechActNorm
                else:
                    speechAct = "rest"
            '''
            speechActNorm = self.normalizemrdatosw(speechAct)
            if len(speechActNorm) != 0 and self.speech_acts_class[speechActNorm] != 0:
                speechAct = speechActNorm
            else:
                if found_S_tag == True:
                    speechAct = 's'
                else:
                    speechAct = "rest"

            return speechAct

    def getTrainAndTestData(self):
        self.trainData = self.data[:int(self.trainPercentage/100 * self.totalDataCount)]
        self.testData = self.data[-int(self.testPercentage/100 * self.totalDataCount):]

    def featurize(self, utterances):
        feature_vectors = []
        speec_acts = []
        utter_text = []
        # form feature vector for sentences
        for utter in utterances:
            feature = Feature(utter)
            #feature_vector = {}
            feature_vector_utter = []
            for headers in feature.featureHeaders:
                #feature_vector[headers] = getattr(feature, headers)()
                feature_vector_utter.append(getattr(feature, headers)())
            speec_acts.append(utter[2])
            utter_text.append(utter[1])
            feature_vectors.append(feature_vector_utter)
            #feature_vectors.append([feature_vector[key] for key in feature_vector])
            #print utter.text, feature_vector

        return feature_vectors, speec_acts, utter_text

    def getData(self):

        for file in listdir(self.datasetPath):
            #print file
            with open(self.datasetPath + '/' + file, 'r') as f:
                utterances = f.readlines()
                for utter in utterances:
                    #print utter
                    utter = utter.split(',')
                    utterID = utter[0]
                    utterText = utter[1]
                    utterSpeechAct = utter[3]
                    #utterPOS = ''
                    #utterPOS = ' '.join([str(x[0] + '/' + x[1]) for x in pos_tag(word_tokenize(utterText))])
                    utterOrginalSpeechAct = sub('\n','',utter[4])
                    utterTokens = word_tokenize(utterText)
                    if utterSpeechAct != utterOrginalSpeechAct:
                            self.misMatchSpeechActCount += 1
                            self.misMatchSpeechAct.append(utter)
                    utterSpeechAct = self.filterSpeechAct(utterSpeechAct)
                    #print utterSpeechAct
                    #sleep(0.1)
                    self.speech_acts_class_count[utterSpeechAct] += 1
                    self.data.append([utterID, utterText, utterSpeechAct, utterTokens])
                    self.totalDataCount += 1
            f.closed

    def normalizeSpeechAct(self, speechActs):
        # normalize speech_acts
        for speechActIndex in range(len(speechActs)):
            if speechActs[speechActIndex] == 's':
                speechActs[speechActIndex] = 'rest'

    def normalizeSpeechActTest(self, speechActs):
        # normalize speech_acts
        for speechActIndex in range(len(speechActs)):
            trimSpeechAct = sub('\^2|\^g|\^m|\^r|\^e|\^q|\^d', '',speechActs[speechActIndex])
            if trimSpeechAct in ['sd', 'sv']:
                speechActs[speechActIndex] = 's'
            elif self.speech_acts_class[speechActs[speechActIndex]] != 1 or \
                    self.speech_acts_class[trimSpeechAct] != 1:
                 speechActs[speechActIndex] = 'rest'

    def normalizePrediction(self, predicted_speech_act, labelledSpeechAct):
        for i in range(len(labelledSpeechAct)):
            if labelledSpeechAct[i] == 's' and predicted_speech_act[i] == 'rest':
                predicted_speech_act[i] = 's'

    '''
    def normalizemrdatosw(self,list_tags):
        for i in range(len(list_tags)):
            tmp = list_tags[i].split('^')
            for j in range(len(tmp)):
                tmp1 = tmp[j].split('|')
            if 's' in tmp1 and 'ar' in tmp1:
                list_tags[i] = 'nn'
            elif 's' in tmp1 and 'aa' in tmp1:
                list_tags[i] = 'ny'
            elif 's' in tmp1 and 'fa' in tmp1:
                list_tags[i] = 'fa'
            elif 's' in tmp1 and 'ft' in tmp1:
                list_tags[i] = 'ft'
        return list_tags
    '''
    def normalizemrdatosw(self,act_tag):
        tag = ''
        tmp = act_tag.split('|')
        tmp1 = []
        for j in range(len(tmp)):
            tmp1.extend(tmp[j].split('^'))
        if 's' in tmp1 and 'ar' in tmp1:
            tag = 'nn'
        elif 's' in tmp1 and 'aa' in tmp1:
            tag = 'ny'
        elif 's' in tmp1 and 'fa' in tmp1:
            tag = 'fa'
        elif 's' in tmp1 and 'ft' in tmp1:
            tag = 'ft'
        elif 's' in tmp1:
            tag = 's'

        return tag

    def combineFeatureVectors(self, feature_vectors_bow, feature_vectors_cust):
        feature_vectors = []
        for i in range(len(feature_vectors_bow)):
            feature_vectors.append(feature_vectors_bow[i] + feature_vectors_cust[i])
        return feature_vectors

    def findmajorityclass(self,speech_act):
        class_dist=Counter(speech_act)
        majority_class=class_dist.most_common(1)
        print "Majority class", majority_class
        count=majority_class[0]
        print "Majority percentage: ",100*count[1]/len(speech_act)

def main():
    '''
    classifier = Classifier('MRDA','../Data/MRDA_Processed/')
    dataStartTime = time()
    classifier.getData()
    dataEndTime = time()
    print "Data loaded in", dataEndTime - dataStartTime, "sec"

    # print classifier.data[2].utterance_count
    # get test and train data
    classifier.getTrainAndTestData()

    featureStartTime = time()
    # transform a feature vector
    feature_vectors, speech_acts, utter_text = classifier.featurize(classifier.trainData)
    featureEndTime = time()
    print "Feature extracted in", featureEndTime - featureStartTime, "sec"
    print len(feature_vectors)

    # normalize speech acts into classes
    #classifier.normalizeSpeechAct(speech_acts)
    print set(speech_acts)
    # train
    trainStartTime = time()
    clf = OneVsRestClassifier(SVC(C=1, kernel = 'linear', gamma= 1, verbose= False, probability=True))
    clf.fit(feature_vectors, speech_acts)
    trainEndTime = time()
    print "Model trained in",trainEndTime - trainStartTime, "sec"

    feature_vectors, labelled_speech_acts, utter_text = classifier.featurize(classifier.testData)

    # normalize speech act for test data
    # classifier.normalizeSpeechAct(labelled_speech_acts)
    print set(labelled_speech_acts)
    # predict speech act for test
    predicted_speech_act = clf.predict(feature_vectors)

    correctResult = Counter()
    wrongResult = Counter()

    for i in range(len(predicted_speech_act)):
        if predicted_speech_act[i] == labelled_speech_acts[i]:
            correctResult[predicted_speech_act[i]] += 1
        else:
            wrongResult[predicted_speech_act[i]] += 1

    total_correct = sum([correctResult[i] for i in correctResult])
    total_wrong = len(predicted_speech_act) - total_correct

    print "total_correct", total_correct
    print "total wrong", total_wrong
    print "accuracy", (total_correct/len(predicted_speech_act)) * 100

    print "Classification_report:\n", classification_report(labelled_speech_acts, predicted_speech_act)#, target_names=target_names)
    print "accuracy_score:", round(accuracy_score(labelled_speech_acts, predicted_speech_act), 2)


    print "correctResult", correctResult

    print "wrongResult", wrongResult
    '''
    # Bag of Words
    classifier = Classifier('MRDA', '../Data/MRDA_Processed/')
    bagofwords = BagOfWords()
    dataStartTime = time()
    classifier.getData()
    dataEndTime = time()
    print "Data loaded in", dataEndTime - dataStartTime, "sec"
    print classifier.speech_acts_class_count
    # print classifier.data[2].utterance_count
    # get test and train data
    classifier.getTrainAndTestData()

    populateSpaceStartTime = time()
    # populate space
    bagofwords.populateSpace(classifier.trainData)
    populateSpaceEndTime = time()
    print "Space populated extracted in", populateSpaceEndTime - populateSpaceStartTime, "sec"
    print "Space length:", len(bagofwords.space)

    #f = open('../Analysis/space.txt','w')
    #f.write(','.join(bagofwords.space))
    #f.close()

    featureStartTime = time()
    # transform a feature vector
    feature_vectors_bow, speech_acts, utter_text = bagofwords.featurize(classifier.trainData)
    featureEndTime = time()
    print "Feature extracted in", featureEndTime - featureStartTime, "sec"
    print len(feature_vectors_bow)

    featureStartTime = time()
    # transform a feature vector
    feature_vectors_cust, speech_acts, utter_text = classifier.featurize(classifier.trainData)
    featureEndTime = time()
    print "Feature extracted in", featureEndTime - featureStartTime, "sec"
    print "feature_vectors_cust",len(feature_vectors_cust)
    feature_vectors = classifier.combineFeatureVectors(feature_vectors_bow, feature_vectors_cust)
    print len(feature_vectors)

    print "set(speech_act) before ", set(speech_acts)
    # normalize speech acts into classes
    classifier.normalizeSpeechAct(speech_acts)
    print "set(speech_act) after ", set(speech_acts)

    classifier.findmajorityclass(speech_acts)
    # train
    trainStartTime = time()
    clf = OneVsRestClassifier(SVC(C=1, kernel = 'linear', gamma = 1, verbose= False, probability=False))
    clf.fit(feature_vectors, speech_acts)
    trainEndTime = time()
    print "Model trained in",trainEndTime - trainStartTime, "sec"

    feature_vectors_bow, labelled_speech_acts, utter_text = bagofwords.featurize(classifier.testData)
    print "len(feature_vectors_bow[0])", len(feature_vectors_bow[0])
    feature_vectors_cust, speech_acts, utter_text = classifier.featurize(classifier.testData)
    print "len(feature_vectors_cust[0])",len(feature_vectors_cust[0])

    feature_vectors = classifier.combineFeatureVectors(feature_vectors_bow, feature_vectors_cust)

    # normalize speech act for test data
    #classifier.normalizeSpeechActTest(labelled_speech_acts)

    predictionStartTime = time()
    # predict speech act for test
    predicted_speech_act = clf.predict(feature_vectors)
    predictionEndTime = time()
    print "Prediction time", predictionEndTime - predictionStartTime

    classifier.normalizePrediction(predicted_speech_act, labelled_speech_acts)
    #Converting MRDA tags to SWDA Tags
    #predicted_speech_act = classifier.normalizemrdatosw(predicted_speech_act)
    #labelled_speech_acts = classifier.normalizemrdatosw(labelled_speech_acts)
    
    print set(predicted_speech_act), set(labelled_speech_acts)
    correctResult = Counter()
    wrongResult = Counter()

    for i in range(len(predicted_speech_act)):
        if predicted_speech_act[i] == labelled_speech_acts[i]:
            correctResult[predicted_speech_act[i]] += 1
        else:
            wrongResult[predicted_speech_act[i]] += 1

    total_correct = sum([correctResult[i] for i in correctResult])
    total_wrong = len(predicted_speech_act) - total_correct

    print "total_correct", total_correct
    print "total wrong", total_wrong
    print "accuracy", (total_correct/len(predicted_speech_act)) * 100

    print "Classification_report:\n", classification_report(labelled_speech_acts, predicted_speech_act)#, target_names=target_names)
    print "accuracy_score:", round(accuracy_score(labelled_speech_acts, predicted_speech_act), 2)

if __name__ == '__main__':
    main()