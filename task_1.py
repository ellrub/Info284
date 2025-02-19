# Importing the default libraries and functions for our models.

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.feature_extraction.text import CountVectorizer
import nltk
nltk.download('wordnet')

import re
from nltk.stem import WordNetLemmatizer

stemmer = WordNetLemmatizer()

data = pd.read_csv('./data/DATASET_REDUX.csv')

# An indenpendent function that cleans the sentences up.
# Returns a list of the cleansed sentences, and the data variable which was used.
def clean_sentences(data):
    all_text = []
    for sen in range(0, len(data)):
        # Remove all the special characters
        text = re.sub(r'\W', ' ', str(data.iloc[sen]))
        # remove all single characters
        text = re.sub(r'\s+[a-zA-Z]\s+', ' ', text)
        # Remove single characters from the start
        text = re.sub(r'\^[a-zA-Z]\s+', ' ', text) 
        # Substituting multiple spaces with single space
        text = re.sub(r'\s+', ' ', text, flags=re.I)
        # Removing prefixed 'b'
        text = re.sub(r'^b\s+', '', text)
        # Converting to Lowercase
        text = text.lower()
        # Lemmatization
        text = text.split()

        text = [stemmer.lemmatize(word) for word in text]
        text = ' '.join(text)

        all_text.append(text)
    return all_text, data  

print(clean_sentences(data))