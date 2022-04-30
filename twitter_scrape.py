import pickle
import time
import tweepy
import oauth2

consumer_key = 'cw30HpFYFPFun7gaI9i3weSKK'
consumer_secret = 'gPJRyk0NphR3gBfH3ZGf1A6O43ghHc66papNjXtzNbaIl4To3P'
access_key = '2613772970-2Li6Gr1DMVjkdXTCJtlv482478b2gC2Ir8L2lKx'
access_secret = '8yNolPTP39MKrBC8r48dsKKAomvk1Ob0bE4aFbgO6aB5l'

auth = tweepy.OAuthHandler(consumer_key, consumer_secret)
auth.set_access_token(access_key, access_secret)
api = tweepy.API(auth)

def limitHandled(cursor):
  while True:
    try:
        yield cursor.next()
    except tweepy.errors.TooManyRequests:
        print("\nSleeping for 1 minute...\n")
        time.sleep(60)
    except StopIteration:
        return

topics = ["comedy","humor","tv","movies","music","hobbies"]
if __name__ == '__main__':
    for topic in topics:
        print(f"Getting tweets from {topic}")
        tweets = []
        for tweet in limitHandled(tweepy.Cursor(api.search_tweets, q=(), count=100, lang='en', tweet_mode="extended").items(5000)):
            if not (hasattr(tweet, "retweeted_status") or \
                    hasattr(tweet, "quoted_status_id"))  or \
                    tweet.in_reply_to_screen_name != None:
                print(f"{topic}\t{tweet.full_text}")
                tweets.append(tweet.full_text.replace("\n"," "))
        filename = "tweets/all.data"
        with open(filename,'a') as f:
            for tweet in tweets:
                f.write(f"{tweet}\n")
        print(f"Tweets from {topic} written.\n")