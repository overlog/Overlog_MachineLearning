import nltk
import pandas as pd
import matplotlib.pyplot as plt

from wordcloud import WordCloud
from nltk.corpus import stopwords
from nltk.tokenize import RegexpTokenizer
from sklearn.cluster import KMeans
from sklearn.metrics import accuracy_score
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import TfidfTransformer

df = pd.DataFrame()

labels = {'UNKNOWN':0, 'ALWAYS':1, 'OKAY':2, 'ERROR':3, 'WARN':4}

with open('logs.txt') as fp:
    line = fp.readline()
    while line:
        line = line.strip().split('|')
        if len(line) > 2:
            label = line[1].strip()
        else:
            label = 'UNKNOWN'
        df = df.append([[line[len(line) - 1], labels[label]]], ignore_index=True)
        line = fp.readline()
df.columns = ['log', 'log_type']

df.to_csv('logs.csv', index=False, encoding='utf-8')

logs = df.log.str.cat(sep=' ')

tokens = RegexpTokenizer(r'\w+').tokenize(logs)
vocabulary = set(tokens)
frequency_dist = nltk.FreqDist(tokens)
#print(sorted(frequency_dist, key=frequency_dist.__getitem__, reverse=True)[0:50])

stop_words = set(stopwords.words('english'))
tokens = [w for w in tokens if not w in stop_words]
frequency_dist = nltk.FreqDist(tokens)
#print(sorted(frequency_dist, key=frequency_dist.__getitem__, reverse=True)[0:50])

'''
word_cloud = WordCloud()
word_cloud.generate_from_frequencies(frequency_dist)

plt.imshow(word_cloud)
plt.axis('off')
plt.show()
'''

X = df.iloc[:, 0]
y = df.iloc[:, 1]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)
vectorizer = TfidfVectorizer()
train_vectors = vectorizer.fit_transform(X_train)
test_vectors = vectorizer.transform(X_test)
#print(train_vectors.shape, test_vectors.shape)

clf = MultinomialNB().fit(train_vectors, y_train)
predicted = clf.predict(test_vectors)
print(accuracy_score(y_test, predicted))
