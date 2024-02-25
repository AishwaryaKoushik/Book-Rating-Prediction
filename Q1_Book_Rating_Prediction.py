#!/usr/bin/env python
# coding: utf-8

# In[23]:


import numpy as np
import pandas as pd
import string
import re
import nltk
from sklearn.feature_extraction.text import CountVectorizer,TfidfVectorizer,TfidfTransformer
import sklearn

# !pip install nltk
# nltk.download('stopwords')


# In[24]:


dataframe = pd.read_csv("Train.csv")

dataframe.head()
dataframe.shape


# In[25]:


dataframe.drop(axis="columns",labels=["user_id","book_id","review_id","date_added","date_updated","read_at","started_at","n_votes","n_comments"],inplace=True)

dataframe.isna().sum()


# Converting it to lowercase

# In[26]:


dataframe['review_text']=dataframe['review_text'].astype('str')


# Lowercase

# In[27]:


dataframe['review_text']=dataframe['review_text'].str.lower()
dataframe.head()


# Expanding contractions

# In[28]:


#!pip install contractions

import contractions
def expand_contractions(text):
    expanded_words = []
    for word in text.split():
       expanded_words.append(contractions.fix(word))
    return ' '.join(expanded_words)


# Removal of Hastags and mentions

# In[29]:


def remove_mentions_and_tags(text):
    text = re.sub(r'@\S*', '', text)
    return re.sub(r'#\S*', '', text)


# Removal of Special characters

# In[30]:


def remove_special_characters(text):
    # define the pattern to keep
    pat = r'[^a-zA-z0-9.,!?/:;\"\'\s]'
    return re.sub(pat, '', text)


# Removal of Punctutations

# In[31]:


PUNCT_TO_REMOVE = string.punctuation
def remove_punctuation(text):
    """custom function to remove the punctuation"""
    return text.translate(str.maketrans('', '', PUNCT_TO_REMOVE))


# Removal of URLs

# In[32]:


def remove_urls(text):
    url_pattern = re.compile(r'https?://\S+|www\.\S+')
    return url_pattern.sub(r'', text)


# Removal of html tags

# In[33]:


def remove_html(text):
    html_pattern = re.compile('<.*?>')
    return html_pattern.sub(r'', text)


# Removal of Stopwords

# In[34]:


from nltk.corpus import stopwords

STOPWORDS = set(stopwords.words('english'))
def remove_stopwords(text):
    """custom function to remove the stopwords"""
    return " ".join([word for word in str(text).split() if word not in STOPWORDS])


# Text Cleaning

# In[35]:


def text_cleaning():
    dataframe['review_text'] = dataframe['review_text'].apply(lambda text: expand_contractions(text))
    dataframe['review_text'] = dataframe['review_text'].apply(lambda text: remove_mentions_and_tags(text))
    dataframe['review_text'] = dataframe['review_text'].apply(lambda text: remove_special_characters(text))
    dataframe['review_text'] = dataframe['review_text'].apply(lambda text: remove_punctuation(text))
    dataframe['review_text'] = dataframe['review_text'].apply(lambda text: remove_urls(text))
    dataframe['review_text'] = dataframe['review_text'].apply(lambda text: remove_html(text))
    dataframe['review_text'] = dataframe['review_text'].apply(lambda text: remove_stopwords(text))

text_cleaning()
dataframe.head()


# Tokenization, Lemmatization and Stemming

# In[36]:


from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer
from nltk.stem import WordNetLemmatizer

stemmer = PorterStemmer()
lemmatizer = WordNetLemmatizer()

def lemmatization_stemming(text):
    tokens = word_tokenize(text)
    stemmed_tokens = [stemmer.stem(word) for word in tokens]
    return " ".join([lemmatizer.lemmatize(word) for word in stemmed_tokens])


# In[37]:


def text_preprocessing():
    dataframe['review_text'] = dataframe['review_text'].apply(lambda text: lemmatization_stemming(text))

dataframe.head()


# Export processed data to csv file

# In[38]:


dataframe.to_csv("preprocessed_data.csv",index=False)


# Test Train Data split

# In[39]:


X= dataframe['review_text']
y= dataframe['rating']

X_train, X_test, y_train, y_test = sklearn.model_selection.train_test_split(X, y, test_size=0.2, random_state=42)
vectorizer = CountVectorizer(analyzer='word',
                             ngram_range=(1, 1),
                             max_features=500,
                             max_df=0.90,
                             min_df=5
                            )

X_train=vectorizer.fit_transform(X_train)
X_test=vectorizer.transform(X_test)

tfidf_transformer = TfidfTransformer(smooth_idf=True, use_idf=True)
X_train = tfidf_transformer.fit_transform(X_train)
X_test = tfidf_transformer.transform(X_test)
X_train = pd.DataFrame(X_train.toarray(),
             columns=vectorizer.get_feature_names_out())
X_test = pd.DataFrame(X_test.toarray(),
             columns=vectorizer.get_feature_names_out())


# Standard Scalar

# In[40]:


from sklearn.preprocessing import StandardScaler

std_scaler = StandardScaler()
X_train=std_scaler.fit_transform(X_train)
X_test=std_scaler.transform(X_test)


# Metrics

# In[41]:


def get_metrics(y_true, y_predict):
  print('accuracy: ', sklearn.metrics.accuracy_score(y_true, y_predict))
  print('F1 Score: ', sklearn.metrics.f1_score(y_true, y_predict, average='micro'))


# Random Forest

# In[42]:


from sklearn.ensemble import RandomForestClassifier

clf = RandomForestClassifier(n_estimators=100, random_state=42)
clf.fit(X_train, y_train)

y_pred = clf.predict(X_test)

get_metrics(y_test,y_pred)


# Logistic Regression

# In[43]:


from sklearn.linear_model import LogisticRegression

logistic_regression=LogisticRegression()
logistic_regression.fit(X_train,y_train)

y_pred=logistic_regression.predict(X_test)

get_metrics(y_test,y_pred)

