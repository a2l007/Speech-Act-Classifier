#!/usr/bin/python
import tweepy
from tweepy import StreamListener
from tweepy import Stream
g=open("source_tweets.txt",'r');
consumer_key='abcd'
consumer_secret='abcd'
access_token='37893525-abcd'
access_token_secret='abcd'
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
