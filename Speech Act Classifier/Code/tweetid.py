#!/usr/bin/python
import tweepy
from tweepy import StreamListener
from tweepy import Stream
g=open("source_tweets.txt",'r');
consumer_key='W0wAfqirB786tf7QxiKxVmtjt'
consumer_secret='2BNI5fY67S0ewjrDhTBlyDXbWhCE7SNmyg9ppNoE3Jp55wty6o'
access_token='37893525-qOxIXuRpVDLXpv0te3mOVqd4IKFcIU68pkanY6xbG'
access_token_secret='GfD200VrffjbUTScuUTWcgeWgXvBo9q6b8mbwpstMQKLM'
auth = tweepy.OAuthHandler(consumer_key, consumer_secret)
auth.set_access_token(access_token, access_token_secret)
api = tweepy.API(auth)
wq=g.readlines()
for l in wq:
	f=open("TweetConvo.txt",'a');
	l.rstrip('\n')
	tid=l.split("\t")[2]
	tid=int(tid)
	tid=[tid]
	stats=api.statuses_lookup(tid)
	parenttweet=[s.in_reply_to_status_id for s in stats]
	stweet=[s.text for s in stats]
	if stweet:
		f.write("\n\n "+str(stweet))
	while parenttweet:
		st=api.statuses_lookup(parenttweet)
		stw=[x.text for x in st]
		if stw:
			f.write(" "+str(stw))
		parenttweet=[i.in_reply_to_status_id for i in st]
	f.close()
g.close()
