from textblob import TextBlob
#from worldcloud import WorldCloud
import numpy as np
import pandas as pd
import re
import matplotlib.pyplot as plt
import seaborn as sns
import nltk
from nltk.sentiment import SentimentIntensityAnalyzer
from tqdm.notebook import tqdm
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn import metrics
# nltk.download('punkt')
# nltk.download('averaged_perceptron_tagger')
# nltk.download('maxent_ne_chunker')
# nltk.download('words')
# nltk.download('vader_lexicon')
plt.style.use('fivethirtyeight')

df_mi_claro = pd.read_csv('df_mi_claro.csv', sep=',')
# print(df_mi_claro.head())
# print(df_mi_claro.columns)
# print(df_mi_claro['content'].head())
# print(df_mi_claro['score'].head())
# print(df_mi_claro['reviewId'].head())
# print(df_mi_claro.shape)
# print(df_mi_claro['score'].value_counts())
# fig, ax = plt.subplots(figsize=(10, 7))
# ax.hist(df_mi_claro['score'], bins = 20)
# ax = df_mi_claro['score'].value_counts()\
#     .sort_index()\
#     .plot(kind='bar', title='Count of Reviews by Stars', figsize=(10, 5))
# ax.set_xlabel('Reviews Stars')
# plt.show()

# Basic NLTK
# example = df_mi_claro['content'][50]
# example_2 = df_mi_claro['content'][51]
# tokens = nltk.word_tokenize(example)
# print(example)
# print(example_2)
# print(tokens)
# Part of Speech tags
# tagged = nltk.pos_tag(tokens)
# print(tagged)
# As a text
# entities = nltk.chunk.ne_chunk(tagged)
# entities.pprint()

# Vader: Valence Aware Dictionary and sEntiment Reasoner

sia = SentimentIntensityAnalyzer()
# print(sia)

#Ejemplos
# example_polarity = sia.polarity_scores(example)
# print(example_polarity)
# example_polarity_2 = sia.polarity_scores(example_2)
# print(example_polarity_2)
# Null values in content column
# print(df_mi_claro.isnull().sum())
# print(df_mi_claro.shape)
# Drop 8 rows where content column is null
df_mi_claro = df_mi_claro[df_mi_claro['content'].notnull()]
# print(df_mi_claro.isnull().sum())
# print(df_mi_claro.shape)
# Run polarity score on the entire dataset

# We save the results in a dictionary
res = {}
for i, row in df_mi_claro.iterrows():
    text = row['content']
    myid = row['reviewId']
    res[myid] = sia.polarity_scores(text)  
vaders = pd.DataFrame(res).T
# print(vaders.head())    
vaders = vaders.reset_index().rename(columns={'index': 'reviewId'})
# vaders.head()
# print(vaders.shape)
vaders = vaders.merge(df_mi_claro, how = 'left')
# vaders.head()
# print(vaders.columns)
vaders_content = vaders[['content', 'neg', 'pos', 'compound']]
vaders_content.to_csv('vaders_content.csv', sep = ',')
# vaders_content.head()

# Plot VADER results
# ax = sns.barplot(data = vaders, x = 'score', y = 'compound')
# ax.set_title('Compound Score by Mi Claro Star Review')
# plt.show()

# Plots of positive, negative, neutral reviews
# fig, axs = plt.subplots(1, 3, figsize=(15, 5))
# sns.barplot(data = vaders, x = 'score', y = 'pos', ax = axs[0])
# sns.barplot(data = vaders, x = 'score', y = 'neg', ax = axs[1])
# sns.barplot(data = vaders, x = 'score', y = 'neu', ax = axs[2])
# axs[0].set_title('Positive')
# axs[1].set_title('Negative')
# axs[2].set_title('Neutral')

# Classification algorithm
classification_starts = vaders[['reviewId', 'neg', 'pos', 'neu', 
                                'compound' ,'content', 'score']]
# classification_starts.head()

X = classification_starts[['neg', 'pos', 'neu', 'compound']]
y = classification_starts['score']
# print(X.head())
# print(y.head())

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3)


rfc = RandomForestClassifier(n_estimators = 100)

rfc.fit(X_train, y_train)
y_pred = rfc.predict(X_test)
# Checking accuracy
print("Accuracy:", metrics.accuracy_score(y_test, y_pred))

