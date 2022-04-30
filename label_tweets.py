'''
Author: Amber Charlotte Converse
File: label_tweets.py
Description: This program is meant to take unlabeled tweet data and
    allow the user to quickly label each tweet manually with a topic.
'''
import random
import os

topics = {1:"news",
          2:"politics/activism",
          3:"popular media",
          4:"video games",
          5:"hobbies",
          6:"humor",
          7:"sports",
          8:"ad",
          9:"personal"}

def main():
    os.chdir("tweet_labelling_project")
    with open("tweets/all_gold.data",'r', encoding="utf-8") as f:
        lines = f.readlines()
        topic_counts = {}
        for line in lines:
            topic = line.split('\t')[0]
            if topic not in topic_counts:
                topic_counts[topic] = 0
            topic_counts[topic] += 1
        print(f"Current size of data is {len(lines)} tweets!")
        for topic, count in topic_counts.items():
            print(f"\t{count} tweets labelled {topic}")
        print()
    with open("tweets/all.data",'r', encoding="utf-8") as f:
        lines = f.readlines()
    indices = list(range(0,len(lines)))
    while len(indices) != 0:
        index = indices[random.randint(0,len(indices)-1)]
        indices.remove(index)
        tweet = lines[index].strip()
        topic = input("Select a topic for this tweet:\n" + \
                      "\t(1) news\n" + \
                      "\t(2) politics/activism\n" + \
                      "\t(3) popular media\n" + \
                      "\t(4) video games\n" + \
                      "\t(5) hobbies\n" + \
                      "\t(6) humor\n" + \
                      "\t(7) sports\n" + \
                      "\t(8) ad\n" + \
                      "\t(9) personal\n" + \
                      tweet + "\n")
        if topic != "":
            try:
                topic = topics[int(topic)]
            except:
                print("\nERROR: Not a valid topic.\n")
                continue
            print(f"\nWriting tweet to gold data file with label {topic}...")
            with open("tweets/all_gold.data",'a', encoding="utf-8") as f:
                f.write(f"{topic}\t{tweet}\n")
            print("Done\n")

if __name__ == "__main__":
    main()