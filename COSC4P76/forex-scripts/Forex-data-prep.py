import json
import datetime
date = datetime.datetime(2007,1,1,0,0,0)
f = open("USDCAD_D1.json","r")
fn = json.load(f)
t = 0
values =[]
test = []
dates =[]
for x in range(len(fn["time"])):
    line = "date , value \n date , value"
    if fn["time"][x]-3682080>= 6311520-1440 and fn["time"][x] - 3682080 <=7364160+2880: # minutes from jan 1 2019 to dec 31st 2020
        t = t+1
        # if (date+datetime.timedelta(days = (fn["time"][x]-3682080)/1440)).year==2019:
        # print(date+datetime.timedelta(days = (fn["time"][x]-3682080)/1440))
        values.append(fn["close"][x])
        dates.append((fn["time"][x]-3682080)/1440)
        if (date + datetime.timedelta(days=(fn["time"][x] - 3682080) / 1440)).year == 2019 and (date + datetime.timedelta(days=(fn["time"][x] - 3682080) / 1440)).month == 5 and (date + datetime.timedelta(days=(fn["time"][x] - 3682080) / 1440)).day == 24:
            dates.append(((fn["time"][x]-3682080)/1440)+2)
            values.append(1.3441)
        # try:
        #     if fn["time"][x+1]-fn["time"][x] >1440:
        #         test.append(fn["time"][x]-fn["time"][x+1])
        #         print(fn["time"][x]-fn["time"][x+1])
        # except:
        #     print(len(test))
        #     pass
# print(len(values))
# print(values)
#####data prep complete######
dates = dates[1:]
diff = [0]
EMA = [values[0]]
multi = 2/628
for x in range(len(values)):
    EMA.append(values[x] * multi + EMA[len(EMA)-1]*(1-multi))
# EMA.remove(0)
actualEMA = EMA[1:]
diffEMA=[0]
for x in range(1,len(actualEMA)):
    diffEMA.append(actualEMA[x]-actualEMA[x-1])
for x in range(1,len(actualEMA)):
    try:
        diff.append(values[x+1]-values[x])
    except:
        continue
# print(values)
# print(diffEMA)
# print(diff)
statement = []
for x in diff:
    if x>0:
        statement.append("UP")
    elif x<0:
        statement.append("DOWN")
    else:
        statement.append("?")

statement = statement[1:len(statement)]
diffEMA = diffEMA[1:len(diffEMA)-1]
# print(len(statement))
# diffEMA = [actualEMA[x]-actualEMA[x-1] for x in range(1,len(actualEMA))]
# print(diffEMA)
line = ""
ndate = datetime.datetime(2019,1,1,0,0,0)
for x in range(len(diffEMA)):
    line = line+str(date+datetime.timedelta(days=dates[x]))+","+str(diffEMA[x])+"\n"
# realdiffEMA = [0]
# print(values)
# print(line)
print(statement)
# for x in diffEMA:
#     realdiffEMA.append(x)
# print(realdiffEMA)
# print(len(EMA))
with open('trainingdata.csv','w+') as outfile:
    # json.dump(diffEMA,outfile)
    outfile.write(line)
with open("label.txt","w+") as output:
    # json.dump(statement,output)
    json.dump(statement,output)