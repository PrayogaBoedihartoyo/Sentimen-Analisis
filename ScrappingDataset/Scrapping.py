import tweepy
import csv #Import csv

####Credentials
consumer_key='5v0ioeaemWDuaCjuo9FeRsbdo'
consumer_secret='kUIqdKXhJFVW4hOa5hwF9dwhYzlo1GKiXmiDuM645Ws0018X4q'
access_token='1187578033365929984-9HlFSjdQ9PueF9FJh8r9rNjLO6owx9'
access_token_secret='21wIxcCIlmgdvmRNH4M25QgAovKsdgarzZUtuEAMm5oKw'

# Authenticate
auth = tweepy.OAuthHandler(consumer_key, consumer_secret)
# Set Tokens
auth.set_access_token(access_token, access_token_secret)

api = tweepy.API(auth)

# Open/create a file to append data to
csvFile = open('DatasetTatapMuka.csv', 'a')

#Use csv writer
csvWriter = csv.writer(csvFile)
header = ["created_at","id_str","username","screen_name"]
csvWriter.writerow(header)
for tweet in tweepy.Cursor(api.search_tweets, q='kuliah luring').items(200):

    # Write a row to the CSV file. I use encode UTF-8
    csvWriter.writerow([tweet.created_at, tweet.id_str, tweet.user.name.encode('utf-8'), tweet.user.screen_name.encode('utf-8'), tweet.text.encode('utf-8')])
    print(tweet.created_at, tweet.text)

csvFile.close()
