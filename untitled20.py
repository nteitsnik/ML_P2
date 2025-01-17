# -*- coding: utf-8 -*-
"""
Created on Mon Jan 13 11:29:31 2025

@author: n.nteits
"""

import kagglehub
import pandas as pd
import numpy as np
import re
from nltk.tokenize import word_tokenize
import nltk
import pandas as pd
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
import os
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn import svm

stemmer = PorterStemmer()
stop_words = set(stopwords.words("english"))

nltk.download('stopwords')

os.environ["OMP_NUM_THREADS"] = "1"

nltk.download('punkt')


nltk.data.path.append(r'C:/nltk_data')



fake_news = pd.read_csv(r'C:\Users\Γιώργος Μπόζιακας\Fake_news_data\Fake.csv')
true_news = pd.read_csv(r'C:\Users\Γιώργος Μπόζιακας\Fake_news_data\True.csv')
fake_df=pd.DataFrame(data=fake_news._data)



fake_news.head(5)
true_news.head(5)

#null check
fake_news.isnull().sum()
true_news.isnull().sum()

#no missing data
fake_news['Class']=1
true_news['Class']=0

#create a unified dataframe
cols=fake_news.columns.values
cols=np.append(cols,'Class')
print(cols)
news=pd.DataFrame(columns=cols)
news=pd.concat([fake_news,true_news])
'''
for i in range(len(news)):
    index = news[news['text'].str.contains('www.', case=False, na=False)].index
'''
news.shape

print(news.groupby('Class').count())


news.iloc[40000]


data=news[['text','Class']]
data.head()
'''
X = news.drop('Class',axis=1)
y = news['Class']
'''
#Clean text

#lowercase
data['text']=data['text'].str.lower()
data = data.reset_index(drop=True)
#remove special characters
for i in range(len(data)):
    
    data.loc[i,'text'] = re.sub(r'[^a-zA-Z0-9\s]', '', data.loc[i,'text'])
    data.loc[i,'text'] = re.sub('\[.*?\]','',data.loc[i,'text'])
    data.loc[i,'text'] = re.sub("\\W"," ",data.loc[i,'text'])
    data.loc[i,'text'] = re.sub('https?://\S+|www\.\S+','',data.loc[i,'text'])
    data.loc[i,'text'] = re.sub('<.*?>+',b'',data.loc[i,'text'])
    #data.loc[i,'text'] = re.sub('[%s]' % re.escape(string.punctuation),'',data.loc[i,'text'])
    data.loc[i,'text'] = re.sub('\w*\d\w*','',data.loc[i,'text'])




textdata=pd.DataFrame(columns=['text','Class'])
for i in range(len(data)):
    textdata.loc[i,'text'] = data.loc[i,'text']
    textdata.loc[i,'Class']=int(data.iloc[i, data.columns.get_loc('Class')])
   
textdata['tokens'] = textdata['text'].apply(lambda x: x.split())

def remove_stopwords(tokens):
    return [word for word in tokens if word not in stop_words]

textdata['tokens'] = textdata['tokens'].apply(remove_stopwords)
   
def text_stemmer(tokens):
    stemmed_tokens = [stemmer.stem(word) for word in tokens]
    return stemmed_tokens
    

textdata['tokens'] = textdata['tokens'].apply(text_stemmer)

#shuffle
textdata = textdata.sample(frac=1, random_state=1).reset_index(drop=True)
textdata['clean_text'] = textdata['tokens'].apply(lambda tokens: ' '.join(tokens))
##Counts
vectorizer = CountVectorizer()
X = vectorizer.fit_transform(textdata['clean_text'])
Y = textdata['Class']
#Browse tokens
feature_names = vectorizer.get_feature_names_out()






#Split Dataset
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.2, random_state=2)


#transform Y_train to numeric
Y_train = np.vectorize(lambda x: pd.to_numeric(x, errors='coerce'))(Y_train)
Y_test = np.vectorize(lambda x: pd.to_numeric(x, errors='coerce'))(Y_test)
##Naive Bayes
nb_model = MultinomialNB(alpha=0.0000001)
nb_model.fit(X_train, Y_train)
##Test
Y_pred = nb_model.predict(X_test)
accuracy = accuracy_score(Y_test, Y_pred)

print(classification_report(Y_test, Y_pred))
print(f"Accuracy: {accuracy * 100:.2f}%")
print(confusion_matrix(Y_test, Y_pred))

tfidf_vectorizer = TfidfVectorizer()
X = tfidf_vectorizer.fit_transform(textdata['clean_text'])
Y = textdata['Class']
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)
Y_train = np.vectorize(lambda x: pd.to_numeric(x, errors='coerce'))(Y_train)
Y_test = np.vectorize(lambda x: pd.to_numeric(x, errors='coerce'))(Y_test)

nb_model = MultinomialNB(alpha=0.00000001)  # alpha=1 for Laplace smoothing
nb_model.fit(X_train, Y_train)
Y_pred = nb_model.predict(X_test)

# Step 5: Evaluate the model
accuracy = accuracy_score(Y_test, Y_pred)
print(f"Accuracy: {accuracy * 100:.2f}%")
print(classification_report(Y_test, Y_pred))
print(classification_report(Y_test, Y_pred))


#svm
SVM = svm.SVC(kernel = 'linear')
SVM.fit(X_train, Y_train)

print("SVM Testing Accuracy Score -> ", SVM.score(X_test, Y_test)*100)
print(confusion_matrix(Y_test, Y_pred))

#Grid search in svm 

vectorizer = CountVectorizer()
X = vectorizer.fit_transform(textdata['clean_text'])
Y = textdata['Class']
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)
Y_train = np.vectorize(lambda x: pd.to_numeric(x, errors='coerce'))(Y_train)
Y_test = np.vectorize(lambda x: pd.to_numeric(x, errors='coerce'))(Y_test)

SVM = svm.SVC(kernel = 'linear')
SVM.fit(X_train, Y_train)

print("SVM Testing Accuracy Score -> ", SVM.score(X_test, Y_test)*100)
print(confusion_matrix(Y_test, Y_pred))


