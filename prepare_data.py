# Importing the default libraries and functions for our models.
import nltk
nltk.download('wordnet')

import re
from nltk.stem import WordNetLemmatizer

stemmer = WordNetLemmatizer()

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
    return all_text

def combine_reviews(data):
    X_1, X_2 = clean_sentences(data["Negative_Review"]), clean_sentences(data["Positive_Review"])
    X = []
    for n in range(len(X_1)):
        X.append(X_1[n] + " " + X_2[n])
    return X

def target_score_to_expression(data):
    y_n = data["Reviewer_Score"].values
    y = []
    for value in y_n:
        if value < 4:
            y.append("negative")
        elif value < 8:
            y.append("neutral")
        else:
            y.append("positive")
        
    return y

def target_score_to_n(data):
    y_n = data["Reviewer_Score"].values
    y = []
    for value in y_n:
        if value < 5:
            y.append(0)
        else:
            y.append(1)
            
    return y

def prepare_nlp(data):
    return combine_reviews(data), target_score_to_expression(data)

def prepare_lstm(data):
    return combine_reviews(data), target_score_to_n(data)

def prepare_knn(data):
    return data

def prepare_cnn(data):
    return data