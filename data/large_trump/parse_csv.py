import pandas as pd
import re
import random

data = pd.read_csv('trump.csv')

data = list(data[data['is_retweet'] == 'false']['text'])

random.shuffle(data)

clean_data = []
for i in range(len(data)):
    # Remove new lines
    try:
        if data[i][0:2] == 'RT' or data[i][0] == '@':
            continue
        clean_data.append(data[i].replace('\n', ''))
    except:
        continue
data = clean_data

f = open('trump_with_links.txt', 'w')
f.writelines('\n'.join(data))
f.close()

for i in range(len(data)):
    # Remove hyperlinks
    data[i] = re.sub(r'\w+:\/{2}[\d\w-]+(\.[\d\w-]+)*(?:(?:\/[^\s/]*))*', '', data[i])

f = open('trump.txt', 'w')
f.writelines('\n'.join(data))
f.close()
