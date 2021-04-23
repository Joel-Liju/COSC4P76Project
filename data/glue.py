with open("2019_Justin_sentiment.csv") as f1:
  a = f1.read().splitlines()

with open("2019_Trump_sentiment.csv") as f2:
  b=f2.read().splitlines()

with open("2020_Justin_sentiment.csv") as f3:
  c=f3.read().splitlines()

with open("2020_Trump_sentiment.csv") as f4:
  d=f4.read().splitlines()

with open("trainingdata.csv") as f5:         
  e=f5.read().splitlines()

with open("label.txt") as f6:        
  import json 
  f=json.loads(f6.read())

for i in range (0,len(a)):
  a[i]=a[i].split(',')
for i in range (0,len(b)):
  b[i]=b[i].split(',')
for i in range (0,len(c)):
  c[i]=c[i].split(',')
for i in range (0,len(d)):
  d[i]=d[i].split(',')
for i in range (0,len(e)):
  e[i]=e[i].split(',')

import csv

with open("OnlyTrump2019.csv", 'w',newline='') as g:
    writer = csv.writer(g)
    writer.writerow(["Trump","class"])
    for i in range (0,len(b)-1):
        writer.writerow([b[i+1][1],f[i]])

with open("OnlyTrudeau2019.csv", 'w',newline='') as g:
    writer = csv.writer(g)
    writer.writerow(["Trudeau","class"])
    for i in range (0,len(a)-1):
        writer.writerow([a[i+1][1],f[i]])

with open("NoTrump2019.csv", 'w',newline='') as g:
    writer = csv.writer(g)
    writer.writerow(["Justin","Moving Average","class"])
    for i in range (0,len(a)-1):
        if  a[i+1][0] !=e[i][0]:
            print("oh noooo ",i)
        writer.writerow([a[i+1][1],e[i][1],f[i]])
with open("NoTrudeau2019.csv", 'w',newline='') as g:
    writer = csv.writer(g)
    writer.writerow(["Trump","Moving Average","class"])
    for i in range (0,len(a)-1):
        if  b[i+1][0] !=e[i][0]:
            print("oh noooo ",i)
        writer.writerow([b[i+1][1],e[i][1],f[i]])

with open("onlyMoving2019.csv", 'w',newline='') as g:
    writer = csv.writer(g)
    writer.writerow(["Moving Average","class"])
    for i in range (0,len(a)-1):
        writer.writerow([e[i][1],f[i]])

with open("complete2019.csv", 'w',newline='') as g:
    writer = csv.writer(g)
    writer.writerow(["Justin","Trump","Moving Average","class"])
    for i in range (0,len(a)-1):
        if a[i+1][0]!=b[i+1][0] or a[i+1][0] !=e[i][0]:
            print("oh noooo ",i)
        writer.writerow([a[i+1][1],b[i+1][1],e[i][1],f[i]])

with open("OnlyTrump2020.csv", 'w',newline='') as g:
    writer = csv.writer(g)
    writer.writerow(["Trump","class"])
    for i in range (0,len(c)-1):
        writer.writerow([c[i+1][1],f[i+313]])

with open("OnlyTrudeau2020.csv", 'w',newline='') as g:
    writer = csv.writer(g)
    writer.writerow(["Trudeau","class"])
    for i in range (0,len(d)-1):
        writer.writerow([d[i+1][1],f[i+313]])

with open("NoTrump2020.csv", 'w',newline='') as g:
    writer = csv.writer(g)
    writer.writerow(["Justin","Moving Average","class"])
    for i in range (0,len(c)-1):
        if  c[i+1][0] !=e[i+313][0]:
            print("oh noooo ",i)
        writer.writerow([c[i+1][1],e[i+313][1],f[i+313]])

with open("NoTrudeau2020.csv", 'w',newline='') as g:
    writer = csv.writer(g)
    writer.writerow(["Trump","Moving Average","class"])
    for i in range (0,len(d)-1):
        if  d[i+1][0] !=e[i+313][0]:
            print("oh noooo ",i)
        writer.writerow([d[i+1][1],e[i+313][1],f[i+313]])

with open("onlyMoving2020.csv", 'w',newline='') as g:
    writer = csv.writer(g)
    writer.writerow(["Moving Average","class"])
    for i in range (0,len(c)-1):
        writer.writerow([e[i+313][1],f[i+313]])

with open("complete2020.csv", 'w',newline='') as g:
    writer = csv.writer(g)
    writer.writerow(["Justin","Trump","Moving Average","class"])
    for i in range (0,len(c)-1):
        if c[i+1][0]!=d[i+1][0] or c[i+1][0] !=e[i+313][0]:
            print("oh noooo ",i)
        writer.writerow([c[i+1][1],d[i+1][1],e[i+313][1],f[i+313]])
