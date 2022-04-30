'''
Author: Amber Charlotte Converse
File: main.py
Description: This file acts as the main for evaluating the
    improvement of the embeddings models over the baseline
    model on correctly labelling tweets.
'''
from typing import Iterator, Iterable, Tuple, Text, Union

import random

import numpy as np
from sklearn import *
from sklearn.metrics import f1_score, accuracy_score
from scipy.sparse import spmatrix

import matplotlib.pyplot as plt

from baseline_model import BaselineModel
from baseline_model import TextToFeaturesBaseline

from elmo_models import ElmoLogModel
from elmo_models import ElmoTFModel
from elmo_models import TextToFeaturesElmo

NDArray = Union[np.ndarray, spmatrix]

tweetLabels = ["news","politics/activism","popular media","video games","hobbies","humor","sports","ad","personal"]

def read_tweets(tweetsPath: str) -> Iterator[Tuple[Text, Text]]:
    '''
    Generates (label, text) tuples from the lines in a tweet file.

    Tweet files contain one message per line. Each line is composed of a label,
    a tab character, and the text of the tweet.

    :param tweetPath: The path of an tweet file, formatted as above.
    :return: An iterator over (label, text) tuples.
    '''
    with open(tweetsPath, 'r') as f:
        labelsTexts = []
        for line in f.readlines():
            line = line.strip().split('\t')
            labelsTexts.append((line[0], line[1]))
    return labelsTexts

class TextToLabels:
    def __init__(self, labels: Iterable[Text]):
        '''
        Initializes an object for converting texts to labels.

        During initialization, the provided training labels are analyzed to
        determine the vocabulary, i.e., all labels that the converter will
        support. Each such label will be associated with a unique integer index
        that may later be accessed via the .index() method.

        :param labels: The training labels.
        '''
        self.labelEncoder = preprocessing.LabelEncoder()
        self.labelEncoder.fit(labels)

    def index(self, label: Text) -> int:
        '''
        Returns the index in the vocabulary of the given label.

        :param label: A label
        :return: The unique integer index associated with the label.
        '''
        return self.labelEncoder.transform([label])[0]

    def __call__(self, labels: Iterable[Text]) -> NDArray:
        '''
        Creates a label vector from a sequence of labels.

        Each entry in the vector corresponds to one of the input labels. The
        value at index j is the unique integer associated with the jth label.

        :param labels: A sequence of labels.
        :return: A vector, with one entry for each label.
        '''
        return self.labelEncoder.transform(labels)

def score_model(model, trainFeatures, trainLabels, testFeatures, testLabels, \
                returnPredictions=False, verbose=False):
    '''
    This function scores a given model using the given feature generator, training
    data, and test data. Accuracy and F1 score will be reported via console output.

    :param model: a class for a trainable model
    :param trainFeatures: pre-created feature matrix for training data
    :param trainLabels: a list of training labels
    :param testFeatures: pre-created feature matrix for text data
    :param testLabels: a list of testing labels
    :param returnPredictions: if this is true, the function returns the list of
        predictions instead of the f1 score (default: False)
    :param verbose: if this is true, the function prints a performance analysis
        of the model

    if returnPredictions:
    :return predictedIndices: the predicted indices on the given test
        data
    :return testIndices: the correct indices for the given test data
    else:
    :return f1: the f1_score of the model
    '''
    toLabels = TextToLabels(trainLabels)

    model = model()
    model.train(trainFeatures, toLabels(trainLabels))

    predictedIndices = model.predict(testFeatures)
    testIndices = toLabels(testLabels)
    f1 = f1_score(testIndices, predictedIndices, average="macro")
    f1s = f1_score(testIndices, predictedIndices, average=None)
    accuracy = accuracy_score(testIndices, predictedIndices)

    if verbose:
        print()
        for label in tweetLabels:
            labelIdx = toLabels.index(label)
            labelF1 = f1s[labelIdx]
            print(f"{labelF1*100:.1f}% F1 for {label} tweets.")
        print(f"\n{f1*100:.1f}% F1 and {accuracy*100:.1f}% accuracy on tweet data.\n")

    if returnPredictions:
        return predictedIndices, testIndices
    else:
        return f1

def compare_models(baselinePredictions, experimentalPredictions, \
        correctIndices, numResamples=10000):
    '''
    This function performs bootstrap resampling on the given predictions from two
    models to determine the p-value which is the chance that the improvement by the
    experimental model happened by chance. The p-value is reported via console
    output.

    :param baselinePredictions: a list of label indices predicted by the baseline
        model.
    :param experimentalPredictions: a list of label indices predicted by the
        experimentalmodel.
    :param correctIndices: a list of label indices which represent the actual
        labels to be compared to each prediction
    :param numResamples: the number of resamples to perform for the p-value
        calculation (default: 10,000)
    '''
    numWorse = 0
    differenceScores = []
    for i in range(len(correctIndices)):
        baselineScore = 1 if baselinePredictions[i] == correctIndices[i] else 0
        experimentalScore = 1 if experimentalPredictions[i] == correctIndices[i] else 0
        differenceScores.append(experimentalScore - baselineScore)
    for i in range(numResamples):
        resample = []
        for i in range(len(differenceScores)):
            resample.append(
                    differenceScores[random.randint(0,len(differenceScores)-1)])
        if sum(resample) <= 0:
            numWorse += 1
    print(f"p={numWorse / numResamples:.20f}")

def run_analysis(trainTexts, trainLabels, testTexts, testLabels, testType):
    '''
    This function runs an analysis of all three models using the
    given training and testing data.

    :param trainTexts: list of strings containing tweets for training
    :param trainLabels: a list of training labels
    :param testTexts: list of strings containing tweets for testing
    :param testLabels: a list of testing labels
    :param testType: a string representing the type of testing data being
        used (generally "Development Tweets" or "Test Tweets"
    '''
    toFeaturesBaseline = TextToFeaturesBaseline(trainTexts)
    toFeaturesElmo = TextToFeaturesElmo()

    testFeaturesBaseline = toFeaturesBaseline(testTexts)
    testFeaturesElmo = toFeaturesElmo(testTexts)

    baselinePredictions, correctIndices = score_model(BaselineModel, \
            toFeaturesBaseline(trainTexts), trainLabels, testFeaturesBaseline, testLabels, True, True)
    elmoLogPredictions, _ = score_model(ElmoLogModel, \
            toFeaturesElmo(trainTexts), trainLabels, testFeaturesElmo, testLabels, True, True)
    elmoTFPredictions, _ = score_model(ElmoTFModel, \
            toFeaturesElmo(trainTexts), trainLabels, testFeaturesElmo, testLabels, True, True)

    compare_models(baselinePredictions, elmoLogPredictions, correctIndices, 100000)
    compare_models(baselinePredictions, elmoTFPredictions, correctIndices, 100000)
    compare_models(elmoLogPredictions, elmoTFPredictions, correctIndices, 100000)

    trainingSizes = list(range(20,len(trainLabels)+1,20))

    f1ScoresBaseline = []
    f1ScoresElmoLog = []
    f1ScoresElmoTF = []

    for numExamples in trainingSizes:
        subsetTrainLabels = []
        subsetTrainTexts = []
        for i in range(numExamples):
            if numExamples - i == 9:
                for label in tweetLabels:
                    j = trainLabels.index(label)
                    subsetTrainLabels.append(trainLabels[j])
                    subsetTrainTexts.append(trainTexts[j])
                break
            else:
                subsetTrainLabels.append(trainLabels[i])
                subsetTrainTexts.append(trainTexts[i])
        f1Baseline = score_model(BaselineModel, \
            toFeaturesBaseline(subsetTrainTexts), subsetTrainLabels, testFeaturesBaseline, testLabels)
        f1ElmoLog = score_model(ElmoLogModel, \
            toFeaturesElmo(subsetTrainTexts), subsetTrainLabels, testFeaturesElmo, testLabels)
        f1ElmoTF = score_model(ElmoTFModel, \
            toFeaturesElmo(subsetTrainTexts), subsetTrainLabels, testFeaturesElmo, testLabels)
        f1ScoresBaseline.append(f1Baseline)
        f1ScoresElmoLog.append(f1ElmoLog)
        f1ScoresElmoTF.append(f1ElmoTF)

    plt.plot(trainingSizes, f1ScoresBaseline, label="Baseline")
    plt.plot(trainingSizes, f1ScoresElmoLog, label="ELMo Logistic")
    plt.plot(trainingSizes, f1ScoresElmoTF, label="ELMo Tensorflow")
    plt.title(f"Learning Curves for Baseline and Experimental Models ({testType})")
    plt.ylabel("F1-Score")
    plt.xlabel("Size of training data (number of tweets)")
    plt.legend(loc=3)

    plt.show()

def main():
    trainLabels, trainTexts = zip(*read_tweets("tweets/tweets.train"))
    develLabels, develTexts = zip(*read_tweets("tweets/tweets.devel"))

    run_analysis(trainTexts, trainLabels, develTexts, develLabels, "Development Tweets")

    # Only use once development is finished
    testLabels, testTexts = zip(*read_tweets("tweets/tweets.test"))

    run_analysis(trainTexts, trainLabels, testTexts, testLabels, "Test Tweets")

if __name__ == "__main__":
    main()