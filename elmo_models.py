'''
Author: Amber Charlotte Converse
File: elmo_models.py
Description: This file contains two of the "fancy" models (both models using ELMo contextual
    word embeddings as features with a logistic regression for weights for one model and
    a 3-layer sequential TensorFlow neural network for the other) for evaluating a tweet's topic.
    I was originally going to use BERT, but I found more accessible information for ELMo.
    Importantly, I was heavily inspired by the following article for the embedding generation:

    https://medium.com/saarthi-ai/elmo-for-contextual-word-embedding-for-text-classification-24c9693b0045
'''
from typing import Iterator, Iterable, Tuple, Text, Union

import re

import numpy as np
from scipy.sparse import spmatrix
import spacy

from sklearn import *
import tensorflow_hub as hub
import tensorflow as tf
import tensorflow.compat.v1 as tf1

nlp = spacy.load("en_core_web_sm")

NDArray = Union[np.ndarray, spmatrix]

class TextToFeaturesElmo:
    def __init__(self):
        '''
        Initializes an object for converting texts to features.
        '''
        tf1.disable_eager_execution()
        self.elmo = hub.Module("https://tfhub.dev/google/elmo/2", trainable=True)

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
        embeddings = self.elmo(texts, signature="default", as_dict=True)["elmo"]

        with tf1.Session() as sess:
            sess.run(tf1.global_variables_initializer())
            sess.run(tf1.tables_initializer())
            return sess.run(tf1.reduce_mean(embeddings,1))

class ElmoLogModel:
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
        '''Trains the classifier using the given training examples.

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

class ElmoTFModel:
    def __init__(self):
        self.model = tf.keras.Sequential([
                tf.keras.layers.Dense(128, activation="relu"),
                tf.keras.layers.Dense(64, activation="relu"),
                tf.keras.layers.Dense(10)])
        self.model.compile(optimizer="adam",
                loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                metrics=["accuracy"])

    def train(self, features: NDArray, labels: NDArray) -> None:
        '''Trains the classifier using the given training examples.

        :param features: A feature matrix, where each row represents a text.
        Such matrices will typically be generated via TextToFeatures.
        :param labels: A label vector, where each entry represents a label.
        '''
        self.model.fit(features, labels, epochs=20)

    def predict(self, features: NDArray) -> NDArray:
        '''
        Makes predictions for each of the given examples.

        :param features: A feature matrix, where each row represents a text.
        :return: A prediction vector, where each entry represents a label.
        '''
        return [np.argmax(prediction) for prediction in self.model.predict(features)]