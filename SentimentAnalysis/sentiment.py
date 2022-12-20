import numpy as np
import pandas as pd
import re
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import RegexpTokenizer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Function to preprocess text
def cleanData(text):
    text = text.lower() # All lowercase

    # tokenize and remove stopwords at same time
    tokens = tokenizer.tokenize(text)
    new_tokens = [token for token in tokens if token not in en_stopwords]
    lemmatized_tokens = [wnl.lemmatize(tokens) for tokens in new_tokens] # lemmatization

    clean_text = " ".join(lemmatized_tokens)

    return clean_text

# Data to be used to train out model
df_amazon = pd.read_csv('data.txt', names=['sentence', 'score'], sep = '\t', engine = 'python')

# Split the dataset into training and test sets
sentences = df_amazon['sentence'].tolist() # Converting the ndarray sentences into a list
scores = df_amazon['score'].tolist() # Get all scores in a list also

X_train, X_test, y_train, y_test = train_test_split(sentences, scores, test_size=0.2, random_state=42)


# Initializing various classes
tokenizer = RegexpTokenizer(r'\w+')
en_stopwords = set(stopwords.words('english'))
wnl = WordNetLemmatizer()
cv = CountVectorizer(ngram_range=(1,2))
svm = SVC()


# Cleaning both out original text and our test text.
clean_train_text = [cleanData(i) for i in X_train]
clean_test_text = [cleanData(i) for i in X_test]


# Vectorization. How many times the word was repeated in the sentence
train_vect = cv.fit_transform(clean_train_text).toarray()
test_vect = cv.transform(clean_test_text).toarray()


## Perform classification. Using SVM

# 1 is positive and 0 is negative
svm.fit(train_vect, y_train)
test_score = svm.predict(test_vect)
  
performance = accuracy_score(y_test, test_score)

print(f"Performance of model is {round((performance*100),4)}%")
