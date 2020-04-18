import pandas as pd
import re
import random

data = pd.read_csv('tweets.csv')

data = data[data['is_retweet'] == False]

hdata = list(data[data['handle'] == 'HillaryClinton']['text'])
tdata = list(data[data['handle'] == 'realDonaldTrump']['text'])

random.shuffle(hdata)
random.shuffle(tdata)

for i in range(len(hdata)):
    # Remove new lines
    hdata[i] = hdata[i].replace('\n', '')

for i in range(len(tdata)):
    # Remove new lines
    tdata[i] = tdata[i].replace('\n', '')

f = open('hillary_with_links.txt', 'w')
f.writelines('\n'.join(hdata))
f.close()

f = open('trump_with_links.txt', 'w')
f.writelines('\n'.join(tdata))
f.close()

for i in range(len(hdata)):
    # Remove hyperlinks
    hdata[i] = re.sub(r'\w+:\/{2}[\d\w-]+(\.[\d\w-]+)*(?:(?:\/[^\s/]*))*', '', hdata[i])

for i in range(len(tdata)):
    # Remove hyperlinks
    tdata[i] = re.sub(r'\w+:\/{2}[\d\w-]+(\.[\d\w-]+)*(?:(?:\/[^\s/]*))*', '', tdata[i])

f = open('hillary.txt', 'w')
f.writelines('\n'.join(hdata))
f.close()

f = open('trump.txt', 'w')
f.writelines('\n'.join(tdata))
f.close()
