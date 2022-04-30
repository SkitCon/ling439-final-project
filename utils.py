'''
Author: Amber Charlotte Converse
File: utils.py
Description: This file contains multiple utility functions which
    were used to set up data for processing and training.
'''
import random
import re
import spacy
nlp = spacy.load("en_core_web_sm")

def create_vocabulary():
    '''
    This function creates a vocabulary file to train the
    ELMo embeddings.
    '''
    with open("tweets/all.data",'r') as f_in:
        vocabulary = set()
        for line in f_in:
            sanitizedLine = re.sub(r'[^a-zA-Z0-9 ]', '', line).lower()
            tokens = [token.text for token in nlp.tokenizer(sanitizedLine)]
            for token in tokens:
                vocabulary.add(token)
    with open("vocabulary.txt",'w') as f_out:
        f_out.write("<S>\n</S>\n<UNK>")
        for token in vocabulary:
            f_out.write(f"\n{token}")

def section_data():
    '''
    This function separates the gold data into training,
    development, and test groups and stores these groups
    into separate files.
    '''
    with open("tweets/all_gold.data",'r') as f:
        tweets = f.readlines()

        totalTweets = len(tweets)
        proportionTrain = 0.4
        proportionDevel = 0.3
        proportionTest = 0.3

    train = []
    for i in range(0,int(totalTweets*proportionTrain)):
        randomIndex = random.randint(0,len(tweets)-1)
        train.append(tweets[randomIndex])
        tweets.pop(randomIndex)
    with open("tweets/tweets.train",'w') as f:
        for tweet in train:
            f.write(tweet)

    devel = []
    for i in range(0,int(totalTweets*proportionDevel)):
        randomIndex = random.randint(0,len(tweets)-1)
        devel.append(tweets[randomIndex])
        tweets.pop(randomIndex)
    with open("tweets/tweets.devel",'w') as f:
        for tweet in devel:
            f.write(tweet)

    test = tweets # remaining tweets go to test
    with open("tweets/tweets.test",'w') as f:
        for tweet in test:
            f.write(tweet)

if __name__ == "__main__":
    section_data()