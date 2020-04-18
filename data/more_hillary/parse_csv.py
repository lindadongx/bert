import pandas as pd
import re
import random

data = pd.read_csv('tweets.csv')

data = list(data[data['Tweet Type'] != 'Retweet']['Text'])

random.shuffle(data)

clean_data = []
for i in range(len(data)):
    # Remove new lines
    try:
        clean_data.append(data[i].replace('\n', ''))
    except:
        continue
data = clean_data

f = open('hillary_with_links.txt', 'w')
f.writelines('\n'.join(data))
f.close()

for i in range(len(data)):
    # Remove hyperlinks
    data[i] = re.sub(r'\w+:\/{2}[\d\w-]+(\.[\d\w-]+)*(?:(?:\/[^\s/]*))*', '', data[i])

f = open('hillary.txt', 'w')
f.writelines('\n'.join(data))
f.close()
