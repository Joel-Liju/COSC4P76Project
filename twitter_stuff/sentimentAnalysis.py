from nltk.sentiment import SentimentIntensityAnalyzer
import datetime
import pandas as pd

sia = SentimentIntensityAnalyzer()
name = "Trump"
file_year = "2020"

df = pd.read_csv(f'./{file_year}_data/{file_year}_{name}_cleaned.csv')
rows, _ = df.shape
tweets = df.TweetText
dates = df.PostDate

tweets_by_date = {}
for i in range(rows):

    score = sia.polarity_scores(tweets[i])
    date = dates[i]
    if date in tweets_by_date:
        tweets_by_date[date].append(score['compound'])
    else:
        tweets_by_date[date] = [score['compound']]


with open(f'./{file_year}_data/results/{file_year}_{name}_sentiment.csv', "w") as f:
    f.write('PostDate,TweetText\n')
    result = []
    for k in tweets_by_date.keys():
        tweets_by_date[k] = sum(tweets_by_date[k])/len(tweets_by_date[k])

        date_split = k.split("-"); yr = int(date_split[0]); month = int(date_split[1]); day = int(date_split[2])
        result.append( [datetime.datetime(yr, month, day), tweets_by_date[k]] )

    result.sort(key=lambda x: x[0])
    for r in result:
        f.write(f'{r[0]},{r[1]}\n')

print(tweets_by_date)



