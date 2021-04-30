import sys

import pandas as pd
import datetime


def is_saturday_or_incorrect_year(date, year):
    # 2020-02-25
    date_split = date.split("-")
    yr = int(date_split[0])
    month = int(date_split[1])
    day = int(date_split[2])

    weekno = datetime.datetime(yr, month, day).weekday()

    if int(year) != yr:
        return True
    if weekno == 5:  # 0=mon, 1=tues, 2=wedn, 3=thurs, 4=frid, 5=sat, 6=sun
        return True


def clean_file(name, year):
    df = pd.read_csv(f'./{year}_data/{name}All_{year}.csv')
    rows, _ = df.shape
    tweets = df.TweetText
    dates = df.PostDate
    print(f'total rows: {rows}')

    with open(f'./{year}_data/{year}_{name}_cleaned.csv', "w", encoding="utf-8") as f:
        f.write("PostDate,TweetText\n")
        for i in range(rows):
            tweet = tweets[i].replace(",", "").replace("\n", "").replace("�", " ").replace("#", "")
            l = int(len(tweet) / 2)

            # removing duplicated tweets
            if tweet[0:l] == tweet[l:]:
                tweet = tweet[0:l]

            date = dates[i]
            date = date.split('T')[0]

            if not is_saturday_or_incorrect_year(date, year):
                f.write(f'{date}, {tweet}\n')


if __name__ == "__main__":
    name = "Justin"
    year = "2020"
    clean_file(name, year)
