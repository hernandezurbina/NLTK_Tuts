from tweepy import Stream
from tweepy import OAuthHandler
from tweepy.streaming import StreamListener
import json
import sentMod as s

#consumer key, consumer secret, access token, access secret.
ckey="sZzy1AvVvX3AYH7VVZontO2WS"
csecret="jSgoD4eAXeU5O12iDFYlgEtMDDQCGLGBZNjcO6FOLZLWhuvVts"
atoken="235363618-QNxpqH3VG81BahTFI5ql9QQfikLNKJDSkSjxoOKy"
asecret="2wdQmUD3h0V75fNP8dtRfMDKq1BZxtYG3MkAn4Y56tQ1U"

class listener(StreamListener):

    def on_data(self, data):
        all_data = json.loads(data)        
        tweet = all_data["text"]
        sentimentVal, confidenceVal = s.sentiment(tweet)
        print(tweet)
        print(sentimentVal, confidenceVal )        
        print("\n\n")
        if (confidenceVal * 100) >= 80:
            with open("twitter-out.txt","a") as f:
                f.write(sentimentVal)
                f.write("\n")            
        return True

    def on_error(self, status):
        print(status)

auth = OAuthHandler(ckey, csecret)
auth.set_access_token(atoken, asecret)
twitterStream = Stream(auth, listener())
twitterStream.filter(track=["happy"])
