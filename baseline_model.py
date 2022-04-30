'''
Author: Amber Charlotte Converse
File: baseline_model.py
Description: This file contains the baseline model (a SAGA logistic
    regression model using unigram and bigram features) for evaluating
    a tweet's topic.
'''
from typing import Iterator, Iterable, Tuple, Text, Union

import numpy as np
from scipy.sparse import spmatrix
from sklearn import *

NDArray = Union[np.ndarray, spmatrix]

class TextToFeaturesBaseline:
    def __init__(self, texts: Iterable[Text]):
        '''
        Initializes an object for converting texts to features.

        During initialization, the provided training texts are analyzed to
        determine the vocabulary, i.e., all feature values that the converter
        will support. Each such feature value will be associated with a unique
        integer index that may later be accessed via the .index() method.

        :param texts: The training texts.
        '''
        self.vectorizer = feature_extraction.text.\
                CountVectorizer(
                                    ngram_range=(1,2),
                                    max_features=4096
                                                        )
        self.vectorizer.fit(texts)

    def index(self, feature: Text):
        '''
        Returns the index in the vocabulary of the given feature value.

        :param feature: A feature
        :return: The unique integer index associated with the feature.
        '''
        features = self.vectorizer.get_feature_names_out()
        return np.where(features==feature)[0][0]

    def __call__(self, texts: Iterable[Text]) -> NDArray:
        '''
        Creates a feature matrix from a sequence of texts.

        Each row of the matrix corresponds to one of the input texts. The value
        at index j of row i is the value in the ith text of the feature
        associated with the unique integer j.

        It is up to the implementer what the value of a feature that is present
        in a text should be, though a common choice is 1. Features that are
        absent from a text will have the value 0.

        :param texts: A sequence of texts.
        :return: A matrix, with one row of feature values for each text.
        '''
        self.vectorizer.get_feature_names_out(texts)
        return self.vectorizer.transform(texts)

class BaselineModel:
    def __init__(self):
        self.logRegr = linear_model.\
                LogisticRegression(
                                    tol=0.00001,
                                    C=0.85,
                                    class_weight="balanced",
                                    solver="saga",
                                    max_iter=16384
                                                            )

    def train(self, features: NDArray, labels: NDArray) -> None:
        '''
        Trains the classifier using the given training examples.

        :param features: A feature matrix, where each row represents a text.
        Such matrices will typically be generated via TextToFeatures.
        :param labels: A label vector, where each entry represents a label.
        '''
        self.logRegr.fit(features, labels)

    def predict(self, features: NDArray) -> NDArray:
        '''
        Makes predictions for each of the given examples.

        :param features: A feature matrix, where each row represents a text.
        :return: A prediction vector, where each entry represents a label.
        '''
        return self.logRegr.predict(features)